"""Index bundle layout, shared by the pipeline (writer) and the host (reader)."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import zstandard

FORMAT = 1
MANIFEST = "manifest.json"
TRANSCRIPT = "transcript.parquet"
SENTENCES_TABLE = "sentences.parquet"
SENTENCES_INDEX = "sentences.faiss"
BUNDLE_NAME = "index.tar.zst"


def pack(src_dir: Path, names: list[str], out: Path, level: int = 19) -> None:
    tmp = out.with_suffix(out.suffix + ".part")
    cctx = zstandard.ZstdCompressor(level=level, threads=-1)
    with (open(tmp, "wb") as fh, cctx.stream_writer(fh) as zfh,
          tarfile.open(fileobj=zfh, mode="w|") as tar):
        for name in names:
            tar.add(src_dir / name, arcname=name, recursive=True)
    tmp.replace(out)


def unpack(bundle: Path, dest: Path) -> None:
    """Extract a bundle; the ``data`` filter rejects absolute paths, ``..`` and links out."""
    dest.mkdir(parents=True, exist_ok=True)
    dctx = zstandard.ZstdDecompressor()
    with open(bundle, "rb") as fh, dctx.stream_reader(fh) as zfh:
        stream = io.BufferedReader(zfh)
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            tar.extractall(dest, filter="data")
