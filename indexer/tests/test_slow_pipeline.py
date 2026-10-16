"""The real pipeline with the tiny models (downloads them): FH_SLOW_TESTS=1."""

import json
import os
import subprocess

import faiss
import numpy as np
import pyarrow.parquet as pq
import pytest
from framehound_indexer import bundle
from framehound_indexer.lock import load_lock

pytestmark = pytest.mark.skipif(not os.environ.get("FH_SLOW_TESTS"),
                                reason="set FH_SLOW_TESTS=1 (downloads tiny models)")


def test_tiny_pipeline_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("FH_MODELS_LOCK", "models.tiny.lock")
    video = tmp_path / "in.mp4"
    parts = ["color=c=red:s=320x180:r=25:d=4", "testsrc=s=320x180:r=25:d=4",
             "color=c=blue:s=320x180:r=25:d=4"]
    args = ["ffmpeg", "-v", "error", "-y"]
    for p in parts:
        args += ["-f", "lavfi", "-i", p]
    args += ["-f", "lavfi", "-i", "sine=frequency=440:duration=12",
             "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
             "-map", "3:a", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(video)]
    subprocess.run(args, check=True)
    lock = load_lock()
    job = {"movie_id": "t", "work_dir": str(tmp_path / "w"), "media": {"path": str(video)},
           "duration_s": 12.0, "models_lock": lock.data, "device": "cpu", "zstd_level": 3}
    (tmp_path / "job.json").write_text(json.dumps(job))
    from framehound_indexer import pipeline

    pipeline.run(tmp_path / "job.json")
    lines = [json.loads(x) for x in (tmp_path / "w/progress.jsonl").read_text().splitlines()]
    assert lines[-1]["stage"] == "done"
    work = tmp_path / "w/work"
    manifest = json.loads((work / bundle.MANIFEST).read_text())
    assert manifest["contents"] == ["dialogue", "visual"]
    assert manifest["counts"]["shots"] == 3 and manifest["counts"]["frames"] == 24

    from framehound_indexer.encoders import encode_text

    index = faiss.read_index(str(work / bundle.CLIP_INDEX))
    t = np.array(pq.read_table(work / bundle.FRAMES).column("t").to_pylist())
    for q, lo, hi in [("a plain red image", 0, 4), ("a plain blue image", 8, 12)]:
        _, ids = index.search(encode_text(q)[None], 3)
        assert all(lo <= t[i] < hi for i in ids[0]), (q, t[ids[0]])
