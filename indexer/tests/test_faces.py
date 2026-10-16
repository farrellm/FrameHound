import math

import numpy as np
import pytest
from framehound_indexer.faces import (
    ARCFACE_DST,
    STRIDES,
    Face,
    align,
    decode,
    invert,
    largest,
    nms,
    normalized_box,
    similarity,
)


def apply(m, p):
    (a, b, c), (d, e, f) = m
    return (a * p[0] + b * p[1] + c, d * p[0] + e * p[1] + f)


def test_similarity_recovers_rotation_scale_and_shift():
    th, s, tx, ty = math.radians(20), 1.7, 12.0, -5.0
    src = list(ARCFACE_DST)
    dst = [(s * (x * math.cos(th) - y * math.sin(th)) + tx,
            s * (x * math.sin(th) + y * math.cos(th)) + ty) for x, y in src]
    m = similarity(src, dst)
    for p, q in zip(src, dst, strict=True):
        assert apply(m, p) == pytest.approx(q, abs=1e-6)
    back = invert(m)
    for p, q in zip(src, dst, strict=True):
        assert apply(back, q) == pytest.approx(p, abs=1e-6)


def test_nms_keeps_the_best_of_overlapping_boxes():
    boxes = [(0, 0, 10, 10), (1, 1, 11, 11), (50, 50, 60, 60)]
    assert nms(boxes, [0.8, 0.9, 0.7]) == [1, 2]


def test_largest():
    small = Face((0, 0, 10, 10), 0.9, [])
    big = Face((0, 0, 30, 20), 0.6, [])
    assert largest([small, big]) is big and largest([]) is None


def scrfd_outputs(size, hits):
    """Synthetic SCRFD outputs (unbatched): ``hits`` maps (stride, anchor index) →
    (score, distances l,t,r,b in strides, 10 point offsets in strides)."""
    scores, dists, kpss = [], [], []
    for stride in STRIDES:
        n = (size // stride) ** 2 * 2
        sc = np.zeros((n, 1), np.float32)
        di = np.zeros((n, 4), np.float32)
        kp = np.zeros((n, 10), np.float32)
        for (st, j), (score, d, k) in hits.items():
            if st == stride:
                sc[j, 0], di[j], kp[j] = score, d, k
        scores.append(sc)
        dists.append(di)
        kpss.append(kp)
    return [*scores, *dists, *kpss]


def test_decode_maps_anchors_to_image_pixels():
    size, stride = 64, 16
    side = size // stride
    # Anchor at grid (x=2, y=1), second anchor of that location.
    j = (1 * side + 2) * 2 + 1
    k = [0.5 * (i % 2 == 0) for i in range(10)]
    out = scrfd_outputs(size, {(stride, j): (0.9, [1, 1, 1, 2], k),
                               (8, 0): (0.3, [1, 1, 1, 1], [0] * 10)})  # below threshold
    faces = decode(out, scale=0.5, size=size, score=0.5, overlap=0.4)
    assert len(faces) == 1
    f = faces[0]
    cx, cy = 2 * stride, 1 * stride
    assert f.box == pytest.approx(((cx - 16) / 0.5, (cy - 16) / 0.5, (cx + 16) / 0.5,
                                   (cy + 32) / 0.5))
    assert f.points[0] == pytest.approx(((cx + 8) / 0.5, cy / 0.5))
    assert f.score == pytest.approx(0.9)


def test_align_puts_landmarks_where_arcface_expects():
    # Draw a dot at each landmark of a face at 2x scale, offset; aligned, the dots
    # land on ARCFACE_DST.
    rgb = np.zeros((300, 300, 3), np.uint8)
    pts = [(x * 2 + 40, y * 2 + 30) for x, y in ARCFACE_DST]
    for x, y in pts:
        rgb[round(y) - 2:round(y) + 3, round(x) - 2:round(x) + 3] = 255
    crop = align(rgb, Face((40, 30, 264, 254), 0.9, pts))
    assert crop.shape == (112, 112, 3)
    for x, y in ARCFACE_DST:
        assert crop[round(y), round(x)].max() > 200


def test_normalized_box_clamps():
    f = Face((-10, 20, 110, 70), 0.9, [])
    assert normalized_box(f, 100, 100) == pytest.approx((0.0, 0.2, 1.0, 0.5))
