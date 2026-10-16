"""Show a picture (SigLIP looks fused with ArcFace identity) and Crop to the face."""

import io
import json
import subprocess

import numpy as np
import pytest
from framehound import db, importer, tasks
from framehound.errors import FHError
from framehound.media import face_crop, ffprobe
from framehound.search.faces import best_per_frame
from framehound.search.visual import fuse
from PIL import Image
from test_pipeline_flow import FAKE, add_local_movie, client, enqueued  # noqa: F401 (fixtures)

FACE_BOX = [0.4, 0.3, 0.1, 0.15]  # the fake index's face 1, in frame 6 (t=3.0 s)


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 100, 50)).save(buf, format="PNG")
    return buf.getvalue()


def ready_movie(data_dir, enqueued, mid="mv1"):  # noqa: F811
    ingest = add_local_movie(data_dir, mid)
    tasks.run_ingest(ingest)
    tasks.run_index(enqueued[-1][1])


def search_image(client, mid="mv1", data=None):  # noqa: F811
    return client.post(f"/movies/{mid}/search", data={"type": "image", "heat_bins": "64"},
                       files={"image": ("q.png", data or png(), "image/png")})


def test_best_per_frame_takes_each_frames_best_face():
    sims = np.array([0.2, 0.9, 0.5, 0.3], dtype=np.float32)
    rows = np.array([0, 2, 2, 5])
    best, which = best_per_frame(sims, rows, 4)
    assert best.tolist() == pytest.approx([0.2, -1, 0.9, -1])
    assert which.tolist() == [0, -1, 1, -1]  # row 5 is past the frames: dropped


def test_fuse_face_matches_clear_the_threshold():
    z_visual = np.array([0.0, 5.0, 1.0, 0.5], dtype=np.float32)
    sims = np.array([-1.0, 0.1, 0.45, 0.39], dtype=np.float32)
    z, from_face = fuse(z_visual, sims, match=0.40, z_min=3.0, bonus=2.0, scale=20.0)
    assert z.tolist() == pytest.approx([0.0, 5.0, 6.0, 0.5])
    assert from_face.tolist() == [False, False, True, False]


@pytest.mark.parametrize(("box", "size", "want"), [
    ([0.45, 0.3, 0.1, 0.15], (1920, 1080), (480, 852, 720, 0)),  # 9:16 fits
    ([0.4, 0.1, 0.2, 0.5], (1920, 1080), (1080, 1080, 420, 0)),  # too big: 1:1, shrunk
    ([0.0, 0.0, 0.05, 0.08], (640, 360), (80, 142, 0, 0)),  # clamped into the corner
])
def test_face_crop(box, size, want):
    w, h, x, y = face_crop(box, *size)
    assert (w, h, x, y) == want
    assert w % 2 == 0 and h % 2 == 0 and x + w <= size[0] and y + h <= size[1]


def test_picture_search_prefers_the_same_face(data_dir, enqueued, client, monkeypatch):  # noqa: F811
    ready_movie(data_dir, enqueued)
    assert client.get("/movies/mv1").json()["capabilities"]["faces"] is True
    from framehound_indexer import encoders

    # Looks match frame 2; the face matches face 1, in frame 6.
    monkeypatch.setattr(encoders, "encode_image", lambda im: np.eye(16, dtype=np.float32)[2])
    monkeypatch.setattr(encoders, "encode_face",
                        lambda im: (np.eye(512, dtype=np.float32)[1], (0.3, 0.3, 0.2, 0.2)))
    r = search_image(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["face_found"] is True and len(body["heat"]) == 64
    top = body["candidates"][0]
    assert top["reason"] == "face" and top["peak_t"] == 3.0
    assert top["face_bbox"] == pytest.approx(FACE_BOX) and top["has_face"] is True
    # (With 8 frames the top-1% threshold sits at the face peak, so looks alone fall short.)
    with db.db() as conn:
        q = db.row(conn, "SELECT * FROM queries WHERE type='image'")
    assert q["image_sha"] and q["q"] is None

    # No face in the picture: looks only, and the UI is told why.
    monkeypatch.setattr(encoders, "encode_face", lambda im: None)
    body = search_image(client).json()
    assert body["face_found"] is False
    assert all(c["reason"] == "visual" for c in body["candidates"])
    assert body["candidates"][0]["peak_t"] == 1.0

    # Describe it marks candidates whose peak frame has a face.
    monkeypatch.setattr(encoders, "encode_text", lambda q: np.eye(16, dtype=np.float32)[6])
    top = client.post("/movies/mv1/search", json={"type": "text", "q": "two people"}
                      ).json()["candidates"][0]
    assert top["reason"] == "visual" and top["has_face"] is True and "face_bbox" not in top


def test_picture_search_rejects_bad_input(data_dir, enqueued, client, monkeypatch):  # noqa: F811
    ready_movie(data_dir, enqueued)
    r = search_image(client, data=b"not an image")
    assert r.status_code == 422 and r.json()["code"] == "bad_image"
    r = client.post("/movies/mv1/search", data={"type": "image"})
    assert r.json()["code"] == "bad_query"
    monkeypatch.setenv("MAX_QUERY_IMAGE_MB", "0.0001")
    from framehound import config

    config.reset_settings()
    r = search_image(client)
    assert r.status_code == 413 and r.json()["code"] == "image_too_big"


def test_cropped_clip(data_dir, enqueued, client):  # noqa: F811
    ready_movie(data_dir, enqueued)
    r = client.post("/movies/mv1/clips", json={"start": 1.0, "end": 3.0, "crop": True,
                                               "mode": "fast"})
    assert r.json()["code"] == "bad_crop"
    r = client.post("/movies/mv1/clips", json={"start": 1.0, "end": 3.0, "crop": True,
                                               "mode": "fast", "face_bbox": FACE_BOX})
    assert r.status_code == 201, r.text
    clip_id = r.json()["clip_id"]
    tasks.run_clip(r.json()["job_id"])
    meta = client.get(f"/clips/{clip_id}").json()
    assert meta["ready"] and meta["crop"] and meta["mode"] == "accurate"
    p = ffprobe(data_dir / "movies/mv1/clips" / f"{clip_id}.mp4")
    # 320x240 source; a 32x36 face → a 90 px square → 9:16 is 90x160.
    assert (p.width, p.height) == face_crop(FACE_BOX, 320, 240)[:2] == (90, 160)


def test_import_rejects_mismatched_face_counts(data_dir):
    from framehound_indexer import bundle, load_lock

    work = data_dir / "fake"
    work.mkdir(parents=True)
    (work / "job.json").write_text(json.dumps(
        {"work_dir": str(work), "stages": ["bundle"], "models_lock": load_lock().data}))
    subprocess.run([*FAKE, str(work / "job.json")], check=True)
    # Claim one face more than the bundle holds.
    staging = work / "re"
    bundle.unpack(work / bundle.BUNDLE_NAME, staging)
    m = json.loads((staging / bundle.MANIFEST).read_text())
    m["counts"]["faces"] += 1
    (staging / bundle.MANIFEST).write_text(json.dumps(m))
    names = [n.name for n in staging.iterdir()]
    bundle.pack(staging, names, work / "bad.tar.zst", level=1)
    with pytest.raises(FHError) as e:
        importer.import_bundle("bad1", work / "bad.tar.zst")
    assert e.value.code == "bad_index" and "face counts" in e.value.detail
