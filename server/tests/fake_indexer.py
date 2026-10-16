"""Stands in for ``python -m framehound_indexer``: writes progress and a real bundle
with a fixed transcript, without loading any models."""

import json
import sys
from pathlib import Path

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from framehound_indexer import bundle
from framehound_indexer.lock import ModelsLock
from framehound_indexer.pipeline import sentence_units

LINES = [(0.2, 1.0, "hello there"), (1.2, 2.6, "never come back here")]


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
    lock = ModelsLock(job["models_lock"])
    (work / bundle.MANIFEST).write_text(json.dumps({
        "format": bundle.FORMAT, "models_lock_hash": lock.hash, "contents": ["dialogue"],
        "counts": {"transcript_segments": len(LINES), "sentences": len(units),
                   "sentences_index": len(units)}}))
    bundle.pack(work, [bundle.MANIFEST, bundle.TRANSCRIPT, bundle.SENTENCES_TABLE,
                       bundle.SENTENCES_INDEX], out / bundle.BUNDLE_NAME, level=1)
    emit(stage="done", state="done", pct=1.0)


if __name__ == "__main__":
    main(sys.argv[1])
