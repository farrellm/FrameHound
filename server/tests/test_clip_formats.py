"""GIF and burned-in subtitle clips, through the real job code and ffmpeg."""

import subprocess

import pytest
from framehound import tasks
from framehound.media import ffprobe
from test_face_search import ready_movie
from test_pipeline_flow import client, enqueued  # noqa: F401 (fixtures)


def test_gif_clip(data_dir, enqueued, client):  # noqa: F811
    ready_movie(data_dir, enqueued)
    r = client.post("/movies/mv1/clips", json={"start": 0.5, "end": 2.5, "format": "gif",
                                               "mode": "fast"})
    assert r.status_code == 201, r.text
    clip_id = r.json()["clip_id"]
    tasks.run_clip(r.json()["job_id"])
    meta = client.get(f"/clips/{clip_id}").json()
    assert meta["ready"] and meta["mode"] == "accurate", meta
    f = client.get(f"/clips/{clip_id}/file")
    assert f.headers["content-type"] == "image/gif" and f.content[:6] == b"GIF89a"
    assert f.headers["content-disposition"].endswith('.gif"')
    p = ffprobe(data_dir / "movies/mv1/clips" / f"{clip_id}.gif")
    assert p.width == 320 and not p.has_audio
    assert p.duration_s == pytest.approx(2.0, abs=0.15)


def test_subtitled_clip(data_dir, enqueued, client):  # noqa: F811
    ready_movie(data_dir, enqueued)
    r = client.post("/movies/mv1/clips", json={"start": 0.0, "end": 3.0, "subs": True,
                                               "mode": "fast"})
    assert r.status_code == 201, r.text
    clip_id = r.json()["clip_id"]
    tasks.run_clip(r.json()["job_id"])
    meta = client.get(f"/clips/{clip_id}").json()
    assert meta["ready"] and meta["subs"] and meta["mode"] == "accurate", meta
    out = data_dir / "movies/mv1/clips" / f"{clip_id}.mp4"
    assert ffprobe(out).duration_s == pytest.approx(3.0, abs=0.1)
    assert not out.with_suffix(".srt").exists()  # written for the cut, then removed

    # The text is burned into the picture: frames under a cue differ from the source.
    def frame(path, t):
        return subprocess.run(["ffmpeg", "-v", "error", "-ss", str(t), "-i", str(path),
                               "-frames:v", "1", "-vf", "crop=iw:ih/4:0:ih*3/4,scale=32:8",
                               "-f", "rawvideo", "-pix_fmt", "gray", "-"],
                              capture_output=True, check=True).stdout

    src = data_dir / "movies/mv1/source.mp4"
    diff = sum(abs(a - b) for a, b in zip(frame(out, 1.8), frame(src, 1.8), strict=True))
    assert diff > 500
