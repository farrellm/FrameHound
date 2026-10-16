"""Query encoders shared by the indexing pipeline and the api.

Everything runs on ONNX Runtime, so the api needs no torch: ``encode_sentence``
(bge-small) and ``encode_text`` / ``encode_image`` (SigLIP 2). The indexing
pipeline uses the same classes, on CUDA when the VM has it. ``encode_face``
arrives with the face stage (M3).
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np

from .lock import ModelsLock, load_lock, models_dir


def snapshot(repo: str, revision: str, patterns: list[str] | None = None) -> Path:
    """Download (or reuse) a pinned model snapshot from the Hugging Face Hub."""
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=repo, revision=revision, allow_patterns=patterns, cache_dir=models_dir()
        )
    )


def materialize(root: Path, files: list[str]) -> Path:
    """A real directory holding ``files`` (hard links where possible).

    The Hub cache stores files as symlinks into ``blobs/``, and onnxruntime refuses
    external weights (``model.onnx_data``) that resolve outside the model's own
    directory.
    """
    if not any((root / f).exists() for f in files[1:]):
        return root
    import shutil

    dest = root.parent.parent / "materialized" / root.name
    for f in files:
        src, dst = root / f, dest / f
        if not src.exists() or dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".tmp")
        tmp.unlink(missing_ok=True)
        try:
            os.link(src.resolve(), tmp)
        except OSError:
            shutil.copyfile(src.resolve(), tmp)
        tmp.replace(dst)
    return dest


class SentenceEncoder:
    """bge-small-en-v1.5: CLS pooling, L2-normalized, 384-d."""

    def __init__(self, lock: ModelsLock | None = None, threads: int | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        cfg = (lock or load_lock()).sentence
        root = snapshot(
            cfg["repo"],
            cfg["revision"],
            [cfg["onnx_file"], "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        )
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=cfg["max_length"])
        self.tokenizer.enable_padding()
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(root / cfg["onnx_file"]), opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.dim = int(cfg["dim"])
        self._lock = threading.Lock()

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self.tokenizer.encode_batch(texts[i : i + batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64),
                "token_type_ids": np.array([e.type_ids for e in enc], dtype=np.int64),
            }
            feeds = {k: v for k, v in feeds.items() if k in self.input_names}
            with self._lock:
                hidden = self.session.run(None, feeds)[0]
            out.append(hidden[:, 0, :])
        vecs = np.concatenate(out).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        return vecs


_sentence: SentenceEncoder | None = None
_sentence_lock = threading.Lock()


def sentence_encoder() -> SentenceEncoder:
    global _sentence
    with _sentence_lock:
        if _sentence is None:
            _sentence = SentenceEncoder()
        return _sentence


def encode_sentence(texts: list[str]) -> np.ndarray:
    return sentence_encoder().encode(texts)


def _normalize(vecs: np.ndarray) -> np.ndarray:
    vecs = vecs.astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
    return vecs


def providers(device: str = "cpu") -> list[str]:
    """ONNX Runtime providers for ``device``; CUDA only when this build has it."""
    import onnxruntime as ort

    if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def preload_cuda() -> None:
    """Make the CUDA and cuDNN libraries findable before a CUDA session is created.

    On Colab they come from torch's pip wheels; importing torch loads them, and
    newer onnxruntime-gpu can load them itself.
    """
    try:
        import torch  # noqa: F401
    except Exception:  # noqa: BLE001, S110 (no torch: rely on the system libraries)
        pass
    try:
        import onnxruntime as ort

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
    except Exception:  # noqa: BLE001, S110
        pass


_RESAMPLE = {"bilinear": 2, "bicubic": 3, "lanczos": 1, "nearest": 0}


class VisualEncoder:
    """SigLIP (2): text and vision towers, each loaded on first use.

    Images are squashed to ``image_size`` square (no crop), rescaled to [0, 1] and
    normalized with the lock's mean/std, as the HF SiglipImageProcessor does. Text
    is lowercased and padded to ``text_max_length``, as SigLIP was trained.
    """

    def __init__(self, lock: ModelsLock | None = None, device: str = "cpu",
                 threads: int | None = None):
        self.cfg = (lock or load_lock()).visual
        self.device = device
        self.threads = threads
        self.dim = int(self.cfg["dim"])
        self.size = int(self.cfg["image_size"])
        self._text = None
        self._vision = None
        self._tokenizer = None
        self._lock = threading.Lock()
        self.mean = np.array(self.cfg["mean"], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array(self.cfg["std"], dtype=np.float32).reshape(1, 3, 1, 1)

    def _root(self, files: list[str]) -> Path:
        pats = [p for f in files for p in (f, f + "_data")]
        return snapshot(self.cfg["repo"], self.cfg["revision"],
                        [*pats, "tokenizer.json", "tokenizer_config.json",
                         "special_tokens_map.json", "preprocessor_config.json"])

    def _session(self, file: str):
        import onnxruntime as ort

        root = materialize(self._root([file]), [file, file + "_data"])
        opts = ort.SessionOptions()
        if self.threads:
            opts.intra_op_num_threads = self.threads
        if self.device == "cuda":
            preload_cuda()
        return ort.InferenceSession(str(root / file), opts, providers=providers(self.device))

    def vision_session(self):
        with self._lock:
            if self._vision is None:
                self._vision = self._session(self.cfg["vision_file"])
            return self._vision

    def text_session(self):
        with self._lock:
            if self._text is None:
                from tokenizers import Tokenizer

                root = self._root([self.cfg["text_file"]])
                tok = Tokenizer.from_file(str(root / "tokenizer.json"))
                n = int(self.cfg["text_max_length"])
                pad = self.cfg["pad_token"]
                tok.enable_truncation(max_length=n)
                tok.enable_padding(length=n, pad_id=tok.token_to_id(pad), pad_token=pad)
                self._tokenizer = tok
                self._text = self._session(self.cfg["text_file"])
            return self._text

    # images ------------------------------------------------------------------

    def preprocess(self, images: list) -> np.ndarray:
        """PIL images or HxWx3 uint8 arrays → NCHW float32 pixel values."""
        from PIL import Image

        resample = _RESAMPLE[self.cfg.get("resample", "bilinear")]
        out = np.empty((len(images), self.size, self.size, 3), dtype=np.uint8)
        for i, im in enumerate(images):
            if isinstance(im, np.ndarray):
                im = Image.fromarray(im)
            im = im.convert("RGB")
            if im.size != (self.size, self.size):
                im = im.resize((self.size, self.size), resample=resample)
            out[i] = np.asarray(im)
        px = out.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
        return (px - self.mean) / self.std

    def encode_pixels(self, pixels: np.ndarray) -> np.ndarray:
        sess = self.vision_session()
        inp = sess.get_inputs()[0]
        dtype = np.float16 if "float16" in inp.type else np.float32
        out = sess.run(["pooler_output"], {inp.name: pixels.astype(dtype)})[0]
        return _normalize(out)

    def encode_images(self, images: list, batch_size: int = 16) -> np.ndarray:
        if not images:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate([self.encode_pixels(self.preprocess(images[i : i + batch_size]))
                               for i in range(0, len(images), batch_size)])

    # text --------------------------------------------------------------------

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        sess = self.text_session()
        if self.cfg.get("lowercase"):
            texts = [t.lower() for t in texts]
        enc = self._tokenizer.encode_batch(texts)
        feeds = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64)}
        names = {i.name for i in sess.get_inputs()}
        if "attention_mask" in names:
            feeds["attention_mask"] = np.array([e.attention_mask for e in enc], dtype=np.int64)
        with self._lock:
            out = sess.run(["pooler_output"], feeds)[0]
        return _normalize(out)


PROMPTS = ["{q}", "a photo of {q}", "a film still of {q}"]


_visual: VisualEncoder | None = None
_visual_lock = threading.Lock()


def visual_encoder() -> VisualEncoder:
    global _visual
    with _visual_lock:
        if _visual is None:
            _visual = VisualEncoder()
        return _visual


def encode_text(q: str) -> np.ndarray:
    """A prompt ensemble: embed each template, average, renormalize (1-d vector)."""
    vecs = visual_encoder().encode_texts([p.format(q=q) for p in PROMPTS])
    return _normalize(vecs.mean(axis=0, keepdims=True))[0]


def encode_image(image) -> np.ndarray:
    return visual_encoder().encode_images([image])[0]
