"""Stands in for ``python -m framehound_indexer``: writes progress and a real bundle
with a fixed transcript and a tiny visual index (8 frames at 2 Hz, two shots),
without loading any models.

Frame embeddings are one-hot over the first ``VISUAL_DIM`` dims, so a query
vector ``e_k`` matches frame k. Faces (``FACES``: frame row, box) are one-hot
over 512 dims, so ``e_j`` matches face j. ``FH_FAKE_STALL`` makes it hang after
starting.
"""

import json
import os
import sys
import time
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from framehound_indexer import bundle
from framehound_indexer.lock import ModelsLock
from framehound_indexer.pipeline import sentence_units

LINES = [(0.2, 1.0, "hello there"), (1.2, 2.6, "never come back here")]
SHOTS = [(0.0, 2.0), (2.0, 4.0)]
FRAMES = 8
VISUAL_DIM = 16
FACES = [(1, (0.1, 0.2, 0.1, 0.15)), (6, (0.4, 0.3, 0.1, 0.15)), (6, (0.7, 0.1, 0.05, 0.08))]


def main(job_path: str) -> None:
    job = json.loads(Path(job_path).read_text())
    out = Path(job["work_dir"])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    prog = open(out / "progress.jsonl", "a", buffering=1)  # noqa: SIM115

    def emit(**kw):
        prog.write(json.dumps({"pct": 0, "eta_s": None, **kw}) + "\n")

    if job.get("fail"):
        emit(stage="error", state="failed", message="boom")
        sys.exit(1)
    if os.environ.get("FH_FAKE_STALL"):
        emit(stage=job["stages"][0], state="running", pct=0.1)
        time.sleep(3600)
    for st in job["stages"]:
        emit(stage=st, state="running", pct=0.5)
        emit(stage=st, state="done", pct=1.0)
    rows = {"seg": [], "start": [], "end": [], "text": [], "words_json": []}
    for i, (a, b, text) in enumerate(LINES):
        ws = text.split()
        step = (b - a) / len(ws)
        rows["seg"].append(i)
        rows["start"].append(a)
        rows["end"].append(b)
        rows["text"].append(text)
        rows["words_json"].append(json.dumps(
            [{"word": w, "start": a + k * step, "end": a + (k + 1) * step} for k, w in enumerate(ws)]))
    pq.write_table(pa.table(rows), work / bundle.TRANSCRIPT)
    units = sentence_units([t for *_, t in LINES])
    pq.write_table(pa.table({"row": list(range(len(units))), "seg_lo": [u[0] for u in units],
                             "seg_hi": [u[1] for u in units]}), work / bundle.SENTENCES_TABLE)
    index = faiss.IndexFlatIP(384)
    index.add(np.eye(len(units), 384, dtype=np.float32))
    faiss.write_index(index, str(work / bundle.SENTENCES_INDEX))
    pq.write_table(pa.table({"idx": [0, 1], "start": [a for a, _ in SHOTS],
                             "end": [b for _, b in SHOTS]}), work / bundle.SHOTS)
    pq.write_table(pa.table({"row": list(range(FRAMES)), "t": [k / 2 for k in range(FRAMES)],
                             "shot_idx": [0 if k < 4 else 1 for k in range(FRAMES)]}),
                   work / bundle.FRAMES)
    clip = faiss.IndexFlatIP(VISUAL_DIM)
    clip.add(np.eye(FRAMES, VISUAL_DIM, dtype=np.float32))
    faiss.write_index(clip, str(work / bundle.CLIP_INDEX))
    (work / bundle.THUMBS).mkdir(exist_ok=True)
    for i in range(len(SHOTS)):
        (work / bundle.THUMBS / f"{i}.jpg").write_bytes(b"\xff\xd8fake\xff\xd9")
    pq.write_table(pa.table({
        "row": pa.array(range(len(FACES)), pa.int32()),
        "frame_row": pa.array([f for f, _ in FACES], pa.int32()),
        **{c: pa.array([b[i] for _, b in FACES], pa.float32())
           for i, c in enumerate(("x", "y", "w", "h"))},
        "det_score": pa.array([0.9] * len(FACES), pa.float32())}), work / bundle.FACES_TABLE)
    faces = faiss.IndexFlatIP(512)
    faces.add(np.eye(len(FACES), 512, dtype=np.float32))
    faiss.write_index(faces, str(work / bundle.FACES_INDEX))
    lock = ModelsLock(job["models_lock"])
    (work / bundle.MANIFEST).write_text(json.dumps({
        "format": bundle.FORMAT, "models_lock_hash": lock.hash,
        "contents": ["dialogue", "visual", "faces"],
        "counts": {"transcript_segments": len(LINES), "sentences": len(units),
                   "sentences_index": len(units), "shots": len(SHOTS), "frames": FRAMES,
                   "clip_index": FRAMES, "faces": len(FACES), "faces_index": len(FACES)}}))
    bundle.pack(work, [bundle.MANIFEST, bundle.TRANSCRIPT, bundle.SENTENCES_TABLE,
                       bundle.SENTENCES_INDEX, bundle.SHOTS, bundle.FRAMES, bundle.CLIP_INDEX,
                       bundle.THUMBS, bundle.FACES_TABLE, bundle.FACES_INDEX],
                out / bundle.BUNDLE_NAME, level=1)
    emit(stage="done", state="done", pct=1.0)


if __name__ == "__main__":
    main(sys.argv[1])
