"""Faces: InsightFace's detector and recognizer, run on ONNX Runtime.

The lock pins two models from InsightFace's ``buffalo_l`` pack: ``det_10g.onnx``
(SCRFD, which finds faces with their eyes, nose tip and mouth corners) and
``w600k_r50.onnx`` (ArcFace, a 512-d identity). The pre- and post-processing
follow insightface's ``model_zoo/scrfd.py``, ``arcface_onnx.py`` and
``utils/face_align.py`` (MIT), ported from degas; the ``insightface`` package
itself needs a compiler, so it isn't used, and alignment uses PIL rather than
OpenCV. The indexer and the api share this code, so both align the same way.
InsightFace's pretrained models are for non-commercial research use.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .lock import ModelsLock, load_lock

Box = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
Point = tuple[float, float]
Affine = tuple[tuple[float, float, float], tuple[float, float, float]]

STRIDES = (8, 16, 32)
ANCHORS = 2  # per location
# SCRFD misses a face that fills the frame, so a query picture with none found is
# tried again smaller (as ComfyUI_IPAdapter_plus does), which shrinks the face
# against the anchors. The index runs at the lock's det_size only.
QUERY_SIZES = (640, 576, 512, 448, 384, 320)
ALIGN = 112

# Where ArcFace expects the eyes, nose tip and mouth corners in a 112 px crop
# (insightface.utils.face_align.arcface_dst).
ARCFACE_DST: tuple[Point, ...] = (
    (38.2946, 51.6963),
    (73.5318, 51.5014),
    (56.0252, 71.7366),
    (41.5493, 92.3655),
    (70.7299, 92.2041),
)


@dataclass(frozen=True)
class Face:
    box: Box
    score: float
    points: list[Point]  # eyes, nose tip, mouth corners

    @property
    def size(self) -> float:
        return min(self.box[2] - self.box[0], self.box[3] - self.box[1])


# geometry ---------------------------------------------------------------------

def similarity(src: Sequence[Point], dst: Sequence[Point]) -> Affine:
    """The rotation, uniform scale and translation that best maps ``src`` onto ``dst``
    (least squares; the same as the Umeyama estimate ``norm_crop`` uses)."""
    n = len(src)
    if n != len(dst) or n < 2:
        raise ValueError("similarity needs two or more matching points")
    sx = sum(p[0] for p in src) / n
    sy = sum(p[1] for p in src) / n
    dx = sum(p[0] for p in dst) / n
    dy = sum(p[1] for p in dst) / n
    norm = dot = cross = 0.0
    for (px, py), (qx, qy) in zip(src, dst, strict=True):
        x, y, u, v = px - sx, py - sy, qx - dx, qy - dy
        norm += x * x + y * y
        dot += x * u + y * v
        cross += x * v - y * u
    if norm == 0:
        raise ValueError("the points coincide")
    a, b = dot / norm, cross / norm
    return ((a, -b, dx - (a * sx - b * sy)), (b, a, dy - (b * sx + a * sy)))


def invert(m: Affine) -> Affine:
    (a, b, c), (d, e, f) = m
    det = a * e - b * d
    if det == 0:
        raise ValueError("singular transform")
    ia, ib, id_, ie = e / det, -b / det, -d / det, a / det
    return ((ia, ib, -(ia * c + ib * f)), (id_, ie, -(id_ * c + ie * f)))


def iou(p: Box, q: Box) -> float:
    """Overlap of two boxes, counting pixels inclusively as InsightFace's NMS does."""
    w = min(p[2], q[2]) - max(p[0], q[0]) + 1
    h = min(p[3], q[3]) - max(p[1], q[1]) + 1
    if w <= 0 or h <= 0:
        return 0.0
    inter = w * h
    area = (p[2] - p[0] + 1) * (p[3] - p[1] + 1) + (q[2] - q[0] + 1) * (q[3] - q[1] + 1)
    return inter / (area - inter)


def nms(boxes: Sequence[Box], scores: Sequence[float], threshold: float = 0.4) -> list[int]:
    """Indices of the boxes kept, best first: each drops the lower-scoring boxes it
    overlaps by more than ``threshold``."""
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    keep: list[int] = []
    for i in order:
        if all(iou(boxes[i], boxes[j]) <= threshold for j in keep):
            keep.append(i)
    return keep


def largest(found: Sequence[Face]) -> Face | None:
    """The biggest face: the one a portrait is of."""
    if not found:
        return None
    return max(found, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))


def decode(outputs: list[Any], scale: float, size: int, score: float,
           overlap: float) -> list[Face]:
    """SCRFD's outputs (scores, box distances and point offsets for each stride) for
    a ``size`` px input, as faces in image pixels, overlapping ones merged."""
    n = len(STRIDES)
    boxes: list[Box] = []
    scores: list[float] = []
    points: list[list[Point]] = []
    for i, stride in enumerate(STRIDES):
        sc, dist, kps = (np.asarray(outputs[i + k * n]) for k in range(3))
        if sc.ndim == 3:  # batched export
            sc, dist, kps = sc[0], dist[0], kps[0]
        side = size // stride
        ys, xs = np.mgrid[:side, :side]
        centres = np.stack([xs, ys], axis=-1).reshape(-1, 2) * stride
        centres = np.repeat(centres, ANCHORS, axis=0).astype(np.float32)
        dist, kps = dist * stride, kps * stride
        for j in np.flatnonzero(sc[:, 0] >= score):
            cx, cy = centres[j]
            d = dist[j]
            boxes.append((float(cx - d[0]) / scale, float(cy - d[1]) / scale,
                          float(cx + d[2]) / scale, float(cy + d[3]) / scale))
            scores.append(float(sc[j, 0]))
            k = kps[j]
            points.append([(float(cx + k[2 * p]) / scale, float(cy + k[2 * p + 1]) / scale)
                           for p in range(5)])
    return [Face(boxes[i], scores[i], points[i]) for i in nms(boxes, scores, overlap)]


def align(rgb: np.ndarray, face: Face, size: int = ALIGN) -> np.ndarray:
    """A ``size`` px crop with the face's eyes, nose and mouth where ArcFace expects them."""
    from PIL import Image

    ratio = size / 112
    dst = [(x * ratio, y * ratio) for x, y in ARCFACE_DST]
    (a, b, c), (d, e, f) = invert(similarity(face.points, dst))
    im = Image.fromarray(rgb)
    out = im.transform((size, size), Image.AFFINE, (a, b, c, d, e, f),
                       resample=Image.BILINEAR, fillcolor=(0, 0, 0))
    return np.asarray(out)


# models -----------------------------------------------------------------------

class FaceEncoder:
    """Finds faces in RGB images (HxWx3 uint8) and turns them into identities."""

    def __init__(self, lock: ModelsLock | None = None, device: str = "cpu",
                 threads: int | None = None, arena: bool = True):
        self.cfg = (lock or load_lock()).faces
        self.device = device
        self.threads = threads
        self.arena = arena
        self.dim = int(self.cfg["dim"])
        self._det = None
        self._rec = None
        self._lock = threading.Lock()

    def _session(self, file: str):
        import onnxruntime as ort

        from .encoders import preload_cuda, providers, session_options, snapshot

        root = snapshot(self.cfg["repo"], self.cfg["revision"], [file])
        opts = session_options(self.threads, self.arena)
        if self.device == "cuda":
            preload_cuda()
        return ort.InferenceSession(str(root / file), opts, providers=providers(self.device))

    def prefetch(self) -> None:
        """Download the models without loading them."""
        from .encoders import snapshot

        snapshot(self.cfg["repo"], self.cfg["revision"],
                 [self.cfg["detector_file"], self.cfg["recognizer_file"]])

    def unload(self) -> bool:
        """Drop both sessions (they reload on next use); True if they were loaded."""
        with self._lock:
            dropped = self._det is not None
            self._det = self._rec = None
            return dropped

    def sessions(self):
        with self._lock:
            if self._det is None:
                self._det = self._session(self.cfg["detector_file"])
                self._rec = self._session(self.cfg["recognizer_file"])
            return self._det, self._rec

    def detect(self, rgb: np.ndarray, size: int | None = None) -> list[Face]:
        """Faces in the image at one detector size, best first."""
        from PIL import Image

        det, _ = self.sessions()
        size = size or int(self.cfg["det_size"])
        h, w = rgb.shape[:2]
        scale = size / max(h, w)
        nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
        canvas = np.zeros((size, size, 3), dtype=np.uint8)
        canvas[:nh, :nw] = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outputs = det.run(None, {det.get_inputs()[0].name: blob})
        return decode(outputs, scale, size, float(self.cfg["det_score"]),
                      float(self.cfg["nms"]))

    def detect_query(self, rgb: np.ndarray) -> list[Face]:
        """Faces in a query picture, retrying smaller when a close-up finds none."""
        for size in QUERY_SIZES:
            if found := self.detect(rgb, size):
                return found
        return []

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        """Aligned 112 px crops → L2-normalized ArcFace identities."""
        if not crops:
            return np.zeros((0, self.dim), dtype=np.float32)
        _, rec = self.sessions()
        name = rec.get_inputs()[0].name
        batch = int(self.cfg.get("batch_size", 64))
        out = []
        for i in range(0, len(crops), batch):
            blob = ((np.stack(crops[i : i + batch]).astype(np.float32) - 127.5) / 127.5)
            out.append(rec.run(None, {name: blob.transpose(0, 3, 1, 2)})[0])
        vecs = np.concatenate(out).astype(np.float32)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
        return vecs


def normalized_box(face: Face, width: int, height: int) -> tuple[float, float, float, float]:
    """The face's box as (x, y, w, h) fractions of the frame, clamped to it."""
    x1 = min(max(face.box[0] / width, 0.0), 1.0)
    y1 = min(max(face.box[1] / height, 0.0), 1.0)
    x2 = min(max(face.box[2] / width, 0.0), 1.0)
    y2 = min(max(face.box[3] / height, 0.0), 1.0)
    return (x1, y1, max(x2 - x1, 0.0), max(y2 - y1, 0.0))
