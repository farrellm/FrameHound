import json

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from framehound import db
from framehound.search.dialogue import DialogueIndex, normalize
from framehound.search.heat import max_pool, nms, robust_z, spans_to_series
from framehound_indexer import bundle
from framehound_indexer.pipeline import sentence_units

LINES = [
    (10.0, "Hello there, how are you today?"),
    (20.0, "I told you never to come back here."),
    (30.0, "Fine weather we are having."),
    (40.0, "Never come back,"),
    (42.0, "do you hear me?"),
    (60.0, "The horse is in the barn."),
]


def words_for(start: float, text: str):
    out, t = [], start
    for w in text.split():
        out.append({"word": w, "start": round(t, 3), "end": round(t + 0.4, 3)})
        t += 0.5
    return out


@pytest.fixture
def movie(data_dir):
    mid = "m1"
    idx = data_dir / "movies" / mid / "index"
    idx.mkdir(parents=True)
    with db.db() as conn:
        for i, (start, text) in enumerate(LINES):
            words = words_for(start, text)
            conn.execute('INSERT INTO transcript_segments VALUES (?,?,?,?,?,?)',
                         (mid, i, start, words[-1]["end"], text, json.dumps(words)))
    units = sentence_units([t for _, t in LINES])
    pq.write_table(pa.table({"row": list(range(len(units))),
                             "seg_lo": [u[0] for u in units], "seg_hi": [u[1] for u in units]}),
                   idx / bundle.SENTENCES_TABLE)
    # Fake embeddings: unit k is basis vector k, so a query vector picks units exactly.
    dim = 32
    vecs = np.eye(len(units), dim, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)
    faiss.write_index(index, str(idx / bundle.SENTENCES_INDEX))
    (idx / bundle.MANIFEST).write_text("{}")
    return DialogueIndex(mid, idx), dim


def test_normalize():
    assert normalize("  “Never   come BACK,” he said!") == "never come back he said"
    assert normalize("Don’t") == "don't"


def test_lexical_finds_line_with_word_bounds(movie):
    idx, _ = movie
    res = idx.search("never to come back", None, 100.0, 5, 64, 3.0)
    top = res["candidates"][0]
    assert top["reason"] == "dialogue" and top["match"] == "lexical"
    assert [w["word"] for w in top["matched_words"]] == ["never", "to", "come", "back"]
    # "I told you never..." starts at 20.0; "never" is the 4th word.
    assert top["start"] == pytest.approx(21.5 - 0.15, abs=1e-3)
    assert top["end"] == pytest.approx(23.4 + 0.15, abs=1e-3)


def test_query_spanning_segments_uses_window(movie):
    idx, _ = movie
    res = idx.search("never come back do you hear me", None, 100.0, 5, 64, 3.0)
    top = res["candidates"][0]
    assert top["start"] < 40.0 and top["end"] > 42.0


def test_short_segment_inside_long_query_is_not_a_hit(movie):
    idx, _ = movie
    # "Never come back," is wholly inside this query; partial_ratio would say 100.
    res = idx.search("never come back to the house on the hill at midnight", None, 100.0, 5,
                     64, 3.0)
    assert all(c["match"] != "lexical" for c in res["candidates"])


def test_semantic_hit_and_heat(movie):
    idx, dim = movie
    q = np.zeros(dim, dtype=np.float32)
    q[5] = 1.0  # unit 5 = "The horse is in the barn."
    res = idx.search("the line about the stable animal", q, 100.0, 5, 100, 3.0)
    top = res["candidates"][0]
    assert top["match"] == "semantic" and top["start"] == pytest.approx(60.0 - 0.15)
    heat = res["heat"]
    assert len(heat) == 100
    assert np.argmax(heat) == 60  # 1 s per bin over 100 s


def test_no_match(movie):
    idx, _ = movie
    res = idx.search("zzqx flibber", None, 100.0, 5, 64, 3.0)
    assert res["candidates"] == []


def test_heat_helpers():
    s = spans_to_series([(1.0, 2.0, 5.0), (1.5, 3.0, 2.0)], 4.0)
    assert list(s) == [0, 0, 5, 5, 2, 2, 0, 0]
    assert max_pool(np.array([0, 0, 9, 0, 0, 0, 0, 1], dtype=np.float32), 4) == [0, 9, 0, 1]
    z = robust_z(np.array([1, 1, 1, 1, 10], dtype=np.float32))
    assert z[-1] > 3
    kept = nms([{"start": 0, "end": 10, "score": 1.0}, {"start": 1, "end": 3, "score": 1.0},
                {"start": 20, "end": 22, "score": 0.5}])
    assert [(k["start"], k["end"]) for k in kept] == [(1, 3), (20, 22)]
