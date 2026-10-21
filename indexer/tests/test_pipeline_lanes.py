import json
import threading
import time

import pytest
from framehound_indexer import encoders, pipeline

STAGES = ["shots", "visual", "faces", "audio", "sentences"]


def run_fake(tmp_path, monkeypatch, device: str, prefetched: list | None = None):
    """Run the lanes with stages that just sleep; return (stage, start, end) spans."""
    spans: list[tuple[str, float, float]] = []
    monkeypatch.setattr(encoders, "prefetch_visual",
                        lambda lock: (prefetched if prefetched is not None else []).append(lock))
    lock = threading.Lock()

    def fake(name):
        def stage(ctx):
            t0 = time.monotonic()
            time.sleep(0.4 if name == "audio" else 0.15)
            with lock:
                spans.append((name, t0, time.monotonic()))
        return stage

    monkeypatch.setattr(pipeline, "STAGES", {s: fake(s) for s in STAGES})
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"work_dir": str(tmp_path / "out"), "stages": STAGES,
                               "device": device}))
    pipeline.run(job)
    return spans


def overlaps(spans, a: str, b: str) -> bool:
    (_, a0, a1), = [s for s in spans if s[0] == a]
    (_, b0, b1), = [s for s in spans if s[0] == b]
    return a0 < b1 and b0 < a1


@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_gpu_stages_take_turns_only_on_a_gpu(tmp_path, monkeypatch, device):
    prefetched: list = []
    spans = run_fake(tmp_path, monkeypatch, device, prefetched)
    assert len(prefetched) == (device == "cuda")
    assert sorted(s[0] for s in spans) == sorted(STAGES)
    assert overlaps(spans, "shots", "audio")  # shot detection still overlaps Whisper
    gpu_overlap = any(overlaps(spans, a, b) for a in ("visual", "faces") for b in ("audio",))
    assert gpu_overlap == (device == "cpu")
    lines = [json.loads(x) for x in (tmp_path / "out" / "progress.jsonl").read_text().splitlines()]
    assert lines[-1]["stage"] == "done"
