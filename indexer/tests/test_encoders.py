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
