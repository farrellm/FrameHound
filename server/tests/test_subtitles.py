"""SRT for burned-in subtitles: shifted to the clip, grouped into cues."""

import itertools

from framehound.subtitles import build_srt, cues, srt_time


def w(word, a, b):
    return {"word": word, "start": a, "end": b}


def test_srt_time():
    assert srt_time(0) == "00:00:00,000"
    assert srt_time(3725.0456) == "01:02:05,046"
    assert srt_time(-0.2) == "00:00:00,000"


def test_shifted_to_the_clip_and_clamped():
    segs = [[w("before", 9.0, 9.5), w("Hello", 10.0, 10.4), w("there", 10.5, 10.9)],
            [w("after", 20.0, 20.5)]]
    assert cues(segs, 10.2, 12.0) == [(0.0, 0.7, "Hello there")]
    srt = build_srt(segs, 10.2, 12.0)
    assert srt == "1\n00:00:00,000 --> 00:00:00,700\nHello there\n\n"


def test_cue_breaks():
    # A pause, then a segment boundary, each start a new cue.
    segs = [[w("one", 0.0, 0.3), w("two", 0.4, 0.7), w("three", 2.0, 2.3)],
            [w("four", 2.4, 2.7)]]
    assert [c[2] for c in cues(segs, 0, 10)] == ["one two", "three", "four"]
    # Overlapping cues end where the next one starts.
    got = cues(segs, 0, 10)
    assert all(a[1] <= b[0] for a, b in itertools.pairwise(got))


def test_long_cue_wraps_to_two_lines_and_splits():
    words = [w(f"word{i:02d}", i * 0.2, i * 0.2 + 0.15) for i in range(30)]
    got = cues([words], 0, 10)
    assert len(got) > 1
    for _, _, text in got:
        lines = text.split("\n")
        assert len(lines) <= 2 and all(len(line) <= 42 for line in lines)
    assert " ".join(t.replace("\n", " ") for *_, t in got).split() == [x["word"] for x in words]


def test_empty_span():
    assert build_srt([[w("hi", 5, 6)]], 0, 2) == ""
