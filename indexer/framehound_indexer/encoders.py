"""Query encoders shared by the indexing pipeline and the api.

M1 ships ``encode_sentence`` (bge-small via ONNX Runtime, no torch).
``encode_text``, ``encode_image`` and ``encode_face`` arrive with the visual
and face stages (M2, M3).
"""

from __future__ import annotations

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
