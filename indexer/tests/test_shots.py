import subprocess
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from framehound_indexer import bundle
from framehound_indexer.shots import adaptive_cuts, content_scores, detect_shots, hsv


def test_hsv_matches_opencv_ranges():
    px = np.array([[[[255, 0, 0], [0, 255, 0], [0, 0, 255], [128, 128, 128]]]], np.uint8)
    h, s, v = hsv(px)[0, 0].T
    assert list(np.round(h)) == [0, 60, 120, 0]
    assert list(np.round(s)) == [255, 255, 255, 0]
    assert list(v) == [255, 255, 255, 128]


def test_adaptive_cuts_ignores_motion_and_keeps_cuts():
    rng = np.random.default_rng(0)
    scores = rng.uniform(1, 4, 300).astype(np.float32)
    scores[100] = 60  # a hard cut
    scores[200:230] = 25 + rng.uniform(0, 3, 30)  # a fast pan: high but sustained
    scores[250] = 40
    scores[252] = 45  # too close to the previous cut (min shot 0.6 s at 25 fps)
    assert adaptive_cuts(scores, fps=25) == [100, 250]


def test_content_scores_continue_across_batches():
    frames = np.zeros((4, 8, 8, 3), np.uint8)
    frames[2:] = 200
    whole, _ = content_scores(frames)
    a, last = content_scores(frames[:2])
    b, _ = content_scores(frames[2:], last)
    assert np.allclose(np.concatenate([a, b]), whole) and whole[2] > 50


def make_cuts_video(path: Path) -> None:
    # Three 2-second shots of different solid colours, with a moving test pattern
    # in the middle one.
    parts = ["color=c=red:s=160x90:r=25:d=2", "testsrc=s=160x90:r=25:d=2",
             "color=c=blue:s=160x90:r=25:d=2"]
    args = ["ffmpeg", "-v", "error", "-y"]
    for p in parts:
        args += ["-f", "lavfi", "-i", p]
    args += ["-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]", "-map", "[v]",
             "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(args, check=True)


def test_detect_shots_on_video(tmp_path):
    video = tmp_path / "cuts.mp4"
    make_cuts_video(video)
    shots = detect_shots(video, {"width": 96, "ratio": 3.0, "min_content": 15.0, "window": 2,
                                 "min_shot_s": 0.6}, 6.0)
    assert len(shots) == 3
    assert [round(a, 1) for a, _ in shots] == [0.0, 2.0, 4.0]
    assert shots[-1][1] == 6.0


def test_visual_stage_with_stub_encoder(tmp_path, monkeypatch):
    """The 2 fps pass: frame rows, shot assignment, thumbnails, index size."""
    from framehound_indexer import encoders, pipeline
    from framehound_indexer.lock import load_lock

    video = tmp_path / "work" / "media.mp4"
    video.parent.mkdir()
    make_cuts_video(video)

    class Stub:
        dim = 4

        def __init__(self, *a, **k):
            pass

        def vision_backend(self):
            return "stub"

        def preprocess(self, images):
            return np.stack([np.asarray(im, np.float32).mean(axis=(0, 1)) for im in images])

        def encode_pixels(self, px):
            v = np.concatenate([px, np.ones((len(px), 1), np.float32)], axis=1)
            return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(encoders, "make_visual_encoder", Stub)
    lock = load_lock("models.tiny.lock")
    ctx = pipeline.Ctx(job={"duration_s": 6.0, "device": "cpu"}, work=video.parent,
                       out=tmp_path, lock=lock, progress=pipeline.Progress(tmp_path / "p.jsonl"))
    pipeline.stage_shots(ctx)
    pipeline.stage_visual(ctx)
    frames = pq.read_table(video.parent / bundle.FRAMES).to_pydict()
    assert len(frames["t"]) == 12
    assert frames["shot_idx"] == [0] * 4 + [1] * 4 + [2] * 4
    assert sorted(p.name for p in (video.parent / bundle.THUMBS).iterdir()) == [
        "0.jpg", "1.jpg", "2.jpg"]
    import faiss

    assert faiss.read_index(str(video.parent / bundle.CLIP_INDEX)).ntotal == 12
