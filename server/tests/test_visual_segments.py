import numpy as np
from framehound.search.heat import robust_z
from framehound.search.visual import moving_max, segments

HZ = 2.0


def series(n: int, bumps: list[tuple[float, float, float]], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    s = rng.uniform(0.04, 0.06, n).astype(np.float32)  # bounded: |z| <= 2
    t = np.arange(n) / HZ
    for a, b, v in bumps:
        s[(t >= a) & (t < b)] = v
    return s


def run(s: np.ndarray, shots=None, z_min=3.0):
    t = np.arange(len(s)) / HZ
    return segments(robust_z(s), t, shots or [], len(s) / HZ, z_min)


def test_finds_and_ranks_bumps():
    s = series(3000, [(100, 104, 0.30), (200, 201, 0.20)])
    segs, thr, _ = run(s)
    assert thr >= 3.0
    assert len(segs) == 2
    first, second = segs
    assert first.peak_z > second.peak_z
    assert 99 <= first.start <= 100 and 104 <= first.end <= 105.5
    assert 100 <= first.peak_t < 104


def test_dropout_is_filled_and_gaps_merge():
    s = series(3000, [(100, 103, 0.3), (103.5, 106, 0.3)])  # one-frame dropout at 103
    segs, _, _ = run(s)
    assert len(segs) == 1 and segs[0].start < 100 and segs[0].end > 106
    # Runs 1.4 s apart merge; 3 s apart don't.
    s = series(3000, [(100, 101, 0.3), (102.4, 103, 0.3), (106, 107, 0.3)])
    assert len(run(s)[0]) == 2


def test_snaps_to_shot_boundaries_within_one_second():
    s = series(3000, [(50.4, 52.6, 0.3)])
    segs, _, _ = run(s, shots=[(0, 50), (50, 53), (53, 200)])
    # snapped to 50 and 53, then padded 0.5 s
    assert segs[0].start == 49.5 and segs[0].end == 53.5
    assert segs[0].core == (50.0, 53.0)


def test_flat_scores_give_no_candidates():
    s = np.full(400, 0.1, np.float32)
    assert run(s)[0] == []


def test_top_percent_threshold():
    # Many strong frames: the threshold rises to the top 1% so only the peak survives.
    s = series(1000, [(0, 200, 0.2), (300, 301, 0.5)])
    segs, thr, _ = run(s)
    assert thr > 3.0 and segs[0].peak_t >= 300


def test_moving_max_window():
    z = np.zeros(10, np.float32)
    z[5] = 1
    t = np.arange(10) / HZ
    assert list(np.flatnonzero(moving_max(z, t))) == [4, 5, 6]


def test_short_run_never_collapses_onto_one_boundary():
    # One strong frame at 89.5 s with a cut at 89.95 s: start and end would both
    # snap to 89.95.
    s = series(3000, [(89.5, 90.0, 0.3)])
    segs, _, _ = run(s, shots=[(0, 89.95), (89.95, 1500)])
    a, b = segs[0].core
    assert a <= 89.5 and b >= 90.0 and b > a
