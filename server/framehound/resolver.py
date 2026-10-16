"""SourceResolver: turn what the user pasted into something downloadable.

archive.org (items and collections), plain HTTP(S), magnet links and
.torrent files. Torrents are downloaded by ``torrent.py`` on the host.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import quote, unquote, urlencode, urlsplit

from .errors import FHError
from .media import VIDEO_EXTS
from .safe_http import SafeClient
from .torrent import info_hash, parse_magnet, parse_torrent, pick_video

CONTENT_TYPE_EXT = {
    "video/mp4": ".mp4", "video/x-m4v": ".m4v", "video/webm": ".webm",
    "video/x-matroska": ".mkv", "video/quicktime": ".mov", "video/x-msvideo": ".avi",
    "video/mpeg": ".mpg", "video/ogg": ".ogv", "video/mp2t": ".ts",
}

# archive.org format names, best first. Within a tier, `source: original`
# wins over derivatives, then the larger file.
ARCHIVE_FORMAT_TIERS = [
    {"h.264", "h.264 hd", "h.264 ia", "mpeg4"},
    {"512kb mpeg4"},
    {"ogg video"},
    {"matroska"},
]

ARCHIVE_HOSTS = {"archive.org", "www.archive.org"}
_DETAILS = re.compile(r"^/(details|download)/([^/]+)(?:/(.+))?$")

UNRECOGNIZED = ("That isn't a web address or magnet link. "
                "Paste a link that starts with https:// or magnet:.")


@dataclass
class ResolvedSource:
    kind: str  # archive | http | torrent
    url: str | None  # a magnet link for torrents (None for a .torrent file)
    filename: str | None  # server-derived, e.g. source.mp4; None until a magnet's metadata
    source_ref: str  # canonical form, used to spot duplicates
    title: str | None = None
    expected_size: int | None = None
    md5: str | None = None


@dataclass
class Collection:
    identifier: str
    title: str
    total: int
    choices: list[dict] = field(default_factory=list)


def classify(text: str) -> str:
    """Cheap syntactic check, shared with the UI's enable rule."""
    t = text.strip()
    if t.lower().startswith("magnet:?"):
        return "magnet"
    parts = urlsplit(t)
    if parts.scheme in ("http", "https") and parts.hostname:
        return "archive" if parts.hostname.lower() in ARCHIVE_HOSTS else "http"
    return "unknown"


class SourceResolver:
    def __init__(self, client: SafeClient | None = None):
        self.client = client or SafeClient()

    def resolve(self, text: str) -> ResolvedSource | Collection:
        kind = classify(text)
        if kind == "magnet":
            return self.resolve_magnet(text.strip())
        if kind == "unknown":
            raise FHError("unrecognized", UNRECOGNIZED)
        if kind == "archive":
            return self.resolve_archive(text.strip())
        return self.resolve_http(text.strip())

    # archive.org ---------------------------------------------------------

    def resolve_archive(self, url: str) -> ResolvedSource | Collection:
        m = _DETAILS.match(urlsplit(url).path)
        if not m:
            raise FHError("unrecognized", "That archive.org link doesn't point to an item. "
                          "Use a link like archive.org/details/…")
        identifier = unquote(m.group(2))
        wanted_file = unquote(m.group(3)) if m.group(1) == "download" and m.group(3) else None
        meta = self.client.get_json(f"https://archive.org/metadata/{quote(identifier)}")
        if not isinstance(meta, dict) or not meta.get("metadata"):
            raise FHError("not_found", "That archive.org item doesn't exist.",
                          detail=f"empty metadata for {identifier}")
        md = meta["metadata"]
        title = _first(md.get("title")) or identifier
        if md.get("mediatype") == "collection":
            return self.collection(identifier, title=title)
        files = meta.get("files") or []
        f = self._pick_file(files, wanted_file)
        if f is None:
            raise FHError("no_video", "This item has no movie file FrameHound can index. "
                          "Try a different item.", detail=f"no video file in {identifier}")
        name = f["name"]
        ext = PurePosixPath(name).suffix.lower()
        return ResolvedSource(
            kind="archive",
            url=f"https://archive.org/download/{quote(identifier)}/{quote(name)}",
            filename=f"source{ext if ext in VIDEO_EXTS else '.mp4'}",
            source_ref=f"archive:{identifier}/{name}",
            title=title,
            expected_size=int(f["size"]) if f.get("size") else None,
            md5=f.get("md5"),
        )

    @staticmethod
    def _pick_file(files: list[dict], wanted: str | None) -> dict | None:
        if wanted:
            for f in files:
                if f.get("name") == wanted and PurePosixPath(wanted).suffix.lower() in VIDEO_EXTS:
                    return f
            return None
        best, best_key = None, None
        for f in files:
            fmt = (f.get("format") or "").lower()
            tier = next((i for i, t in enumerate(ARCHIVE_FORMAT_TIERS) if fmt in t), None)
            if tier is None:
                continue
            key = (-tier, f.get("source") == "original", int(f.get("size") or 0))
            if best_key is None or key > best_key:
                best, best_key = f, key
        return best

    def collection(self, identifier: str, q: str | None = None, title: str | None = None,
                   rows: int = 100) -> Collection:
        query = f"collection:{identifier} AND mediatype:movies"
        if q and q.strip():
            safe = re.sub(r'[\\":()\[\]{}^~*?]', " ", q.strip())
            query += f' AND title:({safe})'
        params = urlencode([("q", query), ("fl[]", "identifier"), ("fl[]", "title"),
                            ("sort[]", "downloads desc"), ("rows", str(rows)),
                            ("output", "json")])
        res = self.client.get_json(f"https://archive.org/advancedsearch.php?{params}")
        resp = (res or {}).get("response", {}) if isinstance(res, dict) else {}
        docs = resp.get("docs") or []
        if title is None:
            meta = self.client.get_json(f"https://archive.org/metadata/{quote(identifier)}")
            title = _first((meta or {}).get("metadata", {}).get("title")) or identifier
        coll = Collection(
            identifier=identifier, title=title, total=int(resp.get("numFound") or 0),
            choices=[{"identifier": d["identifier"],
                      "title": _first(d.get("title")) or d["identifier"],
                      "source": f"https://archive.org/details/{d['identifier']}"}
                     for d in docs if d.get("identifier")],
        )
        if not coll.choices and not q:
            raise FHError("collection_empty", "This collection has no movies FrameHound can "
                          "index. Try a link to a single item.", detail=f"collection {identifier}")
        return coll

    # BitTorrent ------------------------------------------------------------

    @staticmethod
    def resolve_magnet(uri: str) -> ResolvedSource:
        p = parse_magnet(uri)
        return ResolvedSource(kind="torrent", url=uri, filename=None,
                              source_ref=f"btih:{info_hash(p.info_hashes)}",
                              title=_title(p.name))

    @staticmethod
    def resolve_torrent_file(data: bytes) -> ResolvedSource:
        ti = parse_torrent(data)
        f = pick_video(ti)
        if f is None:
            raise FHError("no_video", "This torrent has no movie file FrameHound can index.",
                          detail=f"{ti.num_files()} files, none with a video extension")
        return ResolvedSource(kind="torrent", url=None, filename=f"source{f.ext}",
                              source_ref=f"btih:{info_hash(ti.info_hashes())}",
                              title=_title(ti.name()),
                              expected_size=f.size)

    # plain HTTP ------------------------------------------------------------

    def resolve_http(self, url: str) -> ResolvedSource:
        with self.client.stream(url) as r:
            if r.status_code >= 400:
                raise FHError("unreachable", "FrameHound couldn't download that link "
                              f"(the server said {r.status_code}).", detail=f"HTTP {r.status_code}")
            ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
            size = r.headers.get("content-length")
            final = r.url_final
        path_ext = PurePosixPath(urlsplit(final).path).suffix.lower()
        if ctype.startswith("text/html"):
            raise FHError("not_video", "That link is a web page, not a movie file. "
                          "Paste a direct link to the video.", detail=f"Content-Type {ctype}")
        if ctype.startswith("video/"):
            ext = CONTENT_TYPE_EXT.get(ctype) or (path_ext if path_ext in VIDEO_EXTS else ".mp4")
        elif path_ext in VIDEO_EXTS:
            ext = path_ext
        else:
            raise FHError("not_video", "That link doesn't look like a movie file. "
                          "Paste a direct link to the video.", detail=f"Content-Type {ctype!r}")
        name = unquote(PurePosixPath(urlsplit(final).path).stem) or None
        return ResolvedSource(kind="http", url=final, filename=f"source{ext}", source_ref=url,
                              title=name, expected_size=int(size) if size else None)


def _title(name: str | None) -> str | None:
    """A torrent's name, without a video file's extension."""
    if not name:
        return None
    p = PurePosixPath(name)
    return p.stem if p.suffix.lower() in VIDEO_EXTS else name


def _first(v):
    return v[0] if isinstance(v, list) and v else v
