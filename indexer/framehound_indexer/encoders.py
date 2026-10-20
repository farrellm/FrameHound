"""Query encoders shared by the indexing pipeline and the api.

Everything runs on ONNX Runtime, so the api needs no torch: ``encode_sentence``
(bge-small), ``encode_text`` / ``encode_image`` (SigLIP 2) and ``encode_face``
(InsightFace, in ``faces.py``). The indexing pipeline uses the same classes, on
CUDA when the VM has it; there the SigLIP vision tower runs in PyTorch when
``TORCH_TWINS`` has its checkpoint.
"""

from __future__ import annotations

import os
import threading
import time
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

    def __init__(self, lock: ModelsLock | None = None, threads: int | None = None,
                 arena: bool = True):
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
        opts = session_options(threads, arena)
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
            _sentence = SentenceEncoder(arena=False)
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


def session_options(threads: int | None = None, arena: bool = True):
    """ONNX Runtime options. ``arena=False`` frees activations after each run
    instead of keeping a pool that only grows (worth it for the api's one-off
    queries, not for batch indexing)."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads:
        opts.intra_op_num_threads = threads
    opts.enable_cpu_mem_arena = arena
    return opts


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

# The checkpoints the locked ONNX vision towers were exported from. On CUDA the
# indexer runs these in PyTorch with fused (SDPA) attention, about 1.8x the ONNX
# export's throughput on an L4, with embeddings matching it to cosine 0.99999.
# They are pinned here rather than in models.lock, whose hash every bundle carries.
TORCH_TWINS = {
    ("onnx-community/siglip2-so400m-patch14-384-ONNX", "8ecf3f2bbb135bec33f4bb721c11479bf6a8a254"):
        ("google/siglip2-so400m-patch14-384", "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"),
}


def prefetch_torch_twin(lock: ModelsLock) -> None:
    """Download the vision tower's PyTorch twin (4.3 GB, about 30 s on Colab) ahead
    of the visual stage, which would otherwise wait for it."""
    twin = TORCH_TWINS.get((lock.visual["repo"], lock.visual["revision"]))
    if twin is None:
        return
    try:
        snapshot(twin[0], twin[1], ["config.json", "model.safetensors"])
    except Exception as e:  # noqa: BLE001 (the visual stage retries, or uses ONNX)
        import sys

        print(f"visual: prefetching the PyTorch twin failed ({type(e).__name__}: {e})",
              file=sys.stderr, flush=True)


class VisualEncoder:
    """SigLIP (2): text and vision towers, each loaded on first use.

    Images are squashed to ``image_size`` square (no crop), rescaled to [0, 1] and
    normalized with the lock's mean/std, as the HF SiglipImageProcessor does. Text
    is lowercased and padded to ``text_max_length``, as SigLIP was trained.
    """

    def __init__(self, lock: ModelsLock | None = None, device: str = "cpu",
                 threads: int | None = None, arena: bool = True):
        self.cfg = (lock or load_lock()).visual
        self.device = device
        self.threads = threads
        self.arena = arena
        self.dim = int(self.cfg["dim"])
        self.size = int(self.cfg["image_size"])
        self._text = None
        self._vision = None
        self._torch = None  # the PyTorch vision tower, or False once ruled out
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
        opts = session_options(self.threads, self.arena)
        if self.device == "cuda":
            preload_cuda()
        return ort.InferenceSession(str(root / file), opts, providers=providers(self.device))

    def prefetch(self) -> None:
        """Download both towers without loading them."""
        for f in (self.cfg["text_file"], self.cfg["vision_file"]):
            materialize(self._root([f]), [f, f + "_data"])

    def unload(self, text: bool = False, vision: bool = False) -> bool:
        """Drop the ONNX towers asked for (they reload on next use); True if any was
        loaded. A query already holding one finishes with it."""
        with self._lock:
            dropped = (text and self._text is not None) or (vision and self._vision is not None)
            if text:
                self._text = None
            if vision:
                self._vision = None
            return dropped

    def vision_session(self):
        with self._lock:
            if self._vision is None:
                self._vision = self._session(self.cfg["vision_file"])
            return self._vision

    def torch_vision(self):
        """The vision tower's PyTorch twin on CUDA, or None (no twin, no CUDA torch,
        or it failed to load; the ONNX session is used instead)."""
        with self._lock:
            if self._torch is None:
                self._torch = self._load_torch() or False
            return self._torch or None

    def _load_torch(self):
        twin = TORCH_TWINS.get((self.cfg["repo"], self.cfg["revision"]))
        if self.device != "cuda" or twin is None:
            return None
        try:
            import torch
            from transformers import AutoModel

            if not torch.cuda.is_available():
                return None
            model = AutoModel.from_pretrained(twin[0], revision=twin[1], cache_dir=models_dir(),
                                              dtype=torch.float16, attn_implementation="sdpa")
            return model.vision_model.to("cuda").eval()
        except Exception as e:  # noqa: BLE001 (ONNX still works)
            import sys

            print(f"visual: PyTorch twin unavailable ({type(e).__name__}: {e}); using ONNX",
                  file=sys.stderr, flush=True)
            return None

    def vision_backend(self) -> str:
        """Load the vision tower and say what runs it."""
        if self.torch_vision() is not None:
            return f"torch fp16 sdpa ({TORCH_TWINS[(self.cfg['repo'], self.cfg['revision'])][0]})"
        return f"onnxruntime {self.vision_session().get_providers()}"

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
        model = self.torch_vision()
        if model is not None:
            import torch

            x = torch.from_numpy(pixels).to("cuda", torch.float16)
            with torch.inference_mode():
                return _normalize(model(pixel_values=x).pooler_output.float().cpu().numpy())
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
            _visual = VisualEncoder(arena=False)
        return _visual


# When the api last used each lazily loaded model, for unload_idle.
_used: dict[str, float] = {}


def _touch(name: str) -> None:
    _used[name] = time.monotonic()


def encode_text(q: str) -> np.ndarray:
    """A prompt ensemble: embed each template, average, renormalize (1-d vector)."""
    try:
        vecs = visual_encoder().encode_texts([p.format(q=q) for p in PROMPTS])
    finally:
        _touch("text")
    return _normalize(vecs.mean(axis=0, keepdims=True))[0]


def encode_image(image) -> np.ndarray:
    try:
        return visual_encoder().encode_images([image])[0]
    finally:
        _touch("vision")


_face = None
_face_lock = threading.Lock()


def face_encoder():
    from .faces import FaceEncoder

    global _face
    with _face_lock:
        if _face is None:
            _face = FaceEncoder(arena=False)
        return _face


def encode_face(image) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
    """The largest face in a picture: its ArcFace identity and its box as fractions of
    the picture (x, y, w, h), or None when no face is found."""
    try:
        return _encode_face(image)
    finally:
        _touch("face")


def _encode_face(image):
    from PIL import Image

    from .faces import align, largest, normalized_box

    if not isinstance(image, np.ndarray):
        image = np.asarray(image.convert("RGB") if isinstance(image, Image.Image) else image)
    enc = face_encoder()
    face = largest(enc.detect_query(image))
    if face is None:
        return None
    vec = enc.embed([align(image, face)])[0]
    h, w = image.shape[:2]
    return vec, normalized_box(face, w, h)


def unload_idle(idle_s: float) -> list[str]:
    """Drop the api's models ("text", "vision", "face") unused for ``idle_s``; each
    reloads on the next query that needs it. Returns the names dropped."""
    now = time.monotonic()
    idle = {n for n, t in _used.items() if now - t >= idle_s}
    dropped = []
    if _visual is not None:
        for name in ("text", "vision"):
            if name in idle and _visual.unload(**{name: True}):
                dropped.append(name)
    if "face" in idle and _face is not None and _face.unload():
        dropped.append("face")
    if dropped:
        _trim()
    return dropped


def _trim() -> None:
    """Hand freed memory back to the OS: glibc keeps freed blocks for reuse, so RSS
    wouldn't drop otherwise."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):  # not glibc
        pass
