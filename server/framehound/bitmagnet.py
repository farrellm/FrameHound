"""Find movie torrents by name in a local Bitmagnet (DHT crawler) over GraphQL.

Releases come back grouped under the film Bitmagnet matched them to (TMDB), so
the picker can show one heading per film. Adding a pick goes through the usual
magnet path in the resolver.
"""

from __future__ import annotations

import httpx

from .config import settings
from .errors import FHError

QUERY = """
query Find($q: String!, $limit: Int!) {
  torrentContent {
    search(input: {
      queryString: $q, limit: $limit, totalCount: true,
      facets: { contentType: { filter: [movie] } },
      orderBy: [{ field: relevance, descending: true }]
    }) {
      totalCount
      items {
        infoHash seeders leechers videoResolution videoSource videoCodec
        torrent { name size filesCount magnetUri }
        content { id title releaseYear runtime }
      }
    }
  }
}
"""

UNREACHABLE = "FrameHound can't reach Bitmagnet. Check that it's running on the server."


def search(q: str, limit: int = 40, client: httpx.Client | None = None) -> dict:
    s = settings()
    url = f"{s.bitmagnet_url}/graphql"
    # BITMAGNET_URL is set by whoever runs the server (usually a private Docker
    # address), so this deliberately skips SafeClient's private-network guard.
    own = client is None
    client = client or httpx.Client(timeout=s.bitmagnet_timeout_s)
    try:
        r = client.post(url, json={"query": QUERY, "variables": {"q": q, "limit": limit}})
        r.raise_for_status()
        body = r.json()
    except (httpx.HTTPError, ValueError) as e:
        raise FHError("bitmagnet_unreachable", UNREACHABLE, status=502,
                      detail=f"{url}: {e}") from e
    finally:
        if own:
            client.close()
    if body.get("errors"):
        msgs = "; ".join(str(e.get("message", e)) for e in body["errors"])
        raise FHError("bitmagnet_unreachable", UNREACHABLE, status=502, detail=f"{url}: {msgs}")
    found = (body.get("data") or {}).get("torrentContent", {}).get("search") or {}
    return group(found.get("items") or [], int(found.get("totalCount") or 0), s.max_bytes)


def _release(item: dict, max_bytes: int) -> dict | None:
    t = item.get("torrent") or {}
    if not t.get("magnetUri"):
        return None
    size = t.get("size")
    res = item.get("videoResolution")
    return {
        "info_hash": item.get("infoHash"),
        "magnet": t["magnetUri"],
        "name": t.get("name") or "",
        "size": size,
        "files": t.get("filesCount"),
        "seeders": item.get("seeders"),
        "leechers": item.get("leechers"),
        # Bitmagnet's enum is "V1080p"; people read "1080p".
        "resolution": res[1:] if res and res.startswith("V") else res,
        "source": item.get("videoSource"),
        "codec": item.get("videoCodec"),
        "too_big": bool(size and size > max_bytes),
    }


def group(items: list[dict], total: int, max_bytes: int) -> dict:
    """Films in the order of their best-ranked release; releases by seeders."""
    films: dict[str, dict] = {}
    unmatched: list[dict] = []
    for item in items:
        rel = _release(item, max_bytes)
        if rel is None:
            continue
        c = item.get("content")
        if not c or not c.get("id"):
            unmatched.append(rel)
            continue
        f = films.setdefault(c["id"], {
            "title": c.get("title") or rel["name"], "year": c.get("releaseYear"),
            "runtime_min": c.get("runtime"), "releases": []})
        f["releases"].append(rel)
    for f in films.values():
        f["releases"].sort(key=lambda r: -(r["seeders"] or 0))
    return {"films": list(films.values()), "unmatched": unmatched, "total": total,
            "max_bytes": max_bytes}
