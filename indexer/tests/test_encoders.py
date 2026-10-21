import time

from framehound_indexer import encoders
from framehound_indexer.encoders import VisualEncoder, unload_idle
from framehound_indexer.faces import FaceEncoder


def test_unload_idle_drops_only_idle_models(monkeypatch):
    visual, face = VisualEncoder(), FaceEncoder()
    visual._text, visual._vision = object(), object()
    face._det, face._rec = object(), object()
    monkeypatch.setattr(encoders, "_visual", visual)
    monkeypatch.setattr(encoders, "_face", face)
    now = time.monotonic()
    monkeypatch.setattr(encoders, "_used", {"text": now - 1000, "vision": now, "face": now - 1000})

    assert unload_idle(900) == ["text", "face"]
    assert visual._text is None and visual._vision is not None
    assert face._det is None and face._rec is None
    # Already unloaded: nothing more to drop.
    assert unload_idle(900) == []


def test_unload_idle_before_any_query(monkeypatch):
    monkeypatch.setattr(encoders, "_visual", None)
    monkeypatch.setattr(encoders, "_face", None)
    monkeypatch.setattr(encoders, "_used", {})
    assert unload_idle(0) == []


def test_qwen_size_keeps_aspect_within_bounds():
    from framehound_indexer.encoders import qwen_size

    assert qwen_size(384, 682, 4096, 262144) == (384, 672)  # 16:9 frame: rounds only
    h, w = qwen_size(384, 918, 4096, 262144)  # scope frame: shrinks to fit
    assert h % 32 == 0 and w % 32 == 0 and h * w <= 262144 and abs(w / h - 918 / 384) < 0.1
    h, w = qwen_size(20, 30, 4096, 262144)  # tiny picture: grows to the floor
    assert h * w >= 4096


def test_last_token_takes_each_rows_last_unmasked_state():
    import torch
    from framehound_indexer.encoders import last_token

    hidden = torch.arange(2 * 4 * 1, dtype=torch.float32).reshape(2, 4, 1)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    assert last_token(hidden, mask).flatten().tolist() == [2.0, 7.0]


def test_qwen_lock_changes_only_the_visual_model():
    from framehound_indexer.encoders import QWEN, QwenEncoder, make_visual_encoder, visual_kind
    from framehound_indexer.lock import load_lock

    base, qwen = load_lock(), load_lock("models.qwen.lock")
    assert visual_kind(base) == "siglip" and visual_kind(qwen) == QWEN
    assert {k: v for k, v in base.data.items() if k != "visual"} == \
        {k: v for k, v in qwen.data.items() if k != "visual"}
    assert base.hash != qwen.hash
    # Building the encoder loads nothing; the model loads on first use.
    assert isinstance(make_visual_encoder(qwen), QwenEncoder)
    assert isinstance(make_visual_encoder(base), VisualEncoder)


def test_qwen_model_unloads_only_when_text_and_pictures_are_both_idle(monkeypatch):
    from framehound_indexer.encoders import QwenEncoder
    from framehound_indexer.lock import load_lock

    enc = QwenEncoder(load_lock("models.qwen.lock"))
    enc._model = object()
    monkeypatch.setattr(encoders, "_visual", enc)
    monkeypatch.setattr(encoders, "_face", None)
    now = time.monotonic()
    monkeypatch.setattr(encoders, "_used", {"text": now - 1000, "vision": now})
    assert unload_idle(900) == [] and enc._model is not None
    encoders._used["vision"] = now - 1000
    assert unload_idle(900) == ["text", "vision"] and enc._model is None


def test_encode_text_uses_the_encoders_own_query(monkeypatch):
    import numpy as np

    class Stub:
        def encode_query(self, q):
            return np.array([len(q)], np.float32)

    monkeypatch.setattr(encoders, "_visual", Stub())
    assert encoders.encode_text("abc").tolist() == [3.0]
