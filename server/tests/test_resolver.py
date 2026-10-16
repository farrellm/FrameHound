import httpx
import pytest
from conftest import fixture_json
from framehound.errors import FHError
from framehound.resolver import Collection, ResolvedSource, SourceResolver, classify
from framehound.safe_http import SafeClient

ARCHIVE_IP = "207.241.224.2"


def make_resolver(routes: dict[str, object], seen: list | None = None) -> SourceResolver:
    def handler(req: httpx.Request) -> httpx.Response:
        path = req.url.path
        if seen is not None:
            seen.append(str(req.url))
        for prefix, payload in routes.items():
            if path.startswith(prefix):
                if isinstance(payload, httpx.Response):
                    return payload
                return httpx.Response(200, json=payload)
        return httpx.Response(404, json={})

    client = SafeClient(transport=httpx.MockTransport(handler),
                        resolve=lambda h, p: [ARCHIVE_IP])
    return SourceResolver(client)


def test_classify():
    assert classify("https://archive.org/details/x") == "archive"
    assert classify("magnet:?xt=urn:btih:abc") == "magnet"
    assert classify("https://example.com/a.mp4") == "http"
    assert classify("not a link") == "unknown"


def test_archive_item_picks_best_video():
    r = make_resolver({"/metadata/DuckandC1951": fixture_json("item_duck.json")})
    res = r.resolve("https://archive.org/details/DuckandC1951")
    assert isinstance(res, ResolvedSource)
    # Tier 1 has "h.264 IA" (53 MB) and "MPEG4" (57 MB), both derivatives: larger wins.
    assert res.url == "https://archive.org/download/DuckandC1951/DuckandC1951.mp4"
    assert res.filename == "source.mp4"
    assert res.title == "Duck and Cover"
    assert res.expected_size == 57591076
    assert res.md5


def test_archive_original_beats_derivative_in_tier():
    meta = {"metadata": {"mediatype": "movies", "title": "T"}, "files": [
        {"name": "big.mp4", "format": "h.264", "source": "derivative", "size": "900"},
        {"name": "orig.mp4", "format": "MPEG4", "source": "original", "size": "100"},
        {"name": "low.mp4", "format": "512Kb MPEG4", "source": "original", "size": "5000"},
    ]}
    r = make_resolver({"/metadata/t": meta})
    assert r.resolve("https://archive.org/details/t").url.endswith("/orig.mp4")


def test_archive_download_link_uses_named_file():
    r = make_resolver({"/metadata/DuckandC1951": fixture_json("item_duck.json")})
    res = r.resolve("https://archive.org/download/DuckandC1951/DuckandC1951.ogv")
    assert res.url.endswith("/DuckandC1951.ogv")
    assert res.filename == "source.ogv"


def test_archive_item_without_video():
    r = make_resolver({"/metadata/some_audio": fixture_json("item_no_video.json")})
    with pytest.raises(FHError) as e:
        r.resolve("https://archive.org/details/some_audio")
    assert e.value.code == "no_video"


def test_collection_lists_members():
    seen = []
    r = make_resolver({
        "/metadata/prelinger": {"metadata": {"mediatype": "collection",
                                             "title": "Prelinger Archives"}},
        "/advancedsearch.php": fixture_json("advancedsearch_prelinger.json"),
    }, seen)
    res = r.resolve("https://archive.org/details/prelinger")
    assert isinstance(res, Collection)
    assert res.title == "Prelinger Archives"
    assert res.total == 10461
    assert res.choices[0]["identifier"] == "exercise_lesson_in_health_hygiene"
    assert res.choices[0]["source"] == "https://archive.org/details/exercise_lesson_in_health_hygiene"
    search = next(u for u in seen if "advancedsearch" in u)
    assert "collection%3Aprelinger+AND+mediatype%3Amovies" in search


def test_collection_with_no_videos():
    # archive.org/details/bittorrent ("Archive Torrents") has no movie members.
    r = make_resolver({
        "/metadata/bittorrent": fixture_json("collection_bittorrent.json"),
        "/advancedsearch.php": {"response": {"numFound": 0, "docs": []}},
    })
    with pytest.raises(FHError) as e:
        r.resolve("https://archive.org/details/bittorrent")
    assert e.value.code == "collection_empty"


def test_http_video_and_html():
    r = make_resolver({
        "/movie": httpx.Response(200, headers={"content-type": "video/webm",
                                               "content-length": "1234"}),
        "/page": httpx.Response(200, headers={"content-type": "text/html"}),
    })
    res = r.resolve("https://example.com/movie")
    assert (res.kind, res.filename, res.expected_size) == ("http", "source.webm", 1234)
    with pytest.raises(FHError) as e:
        r.resolve("https://example.com/page")
    assert e.value.code == "not_video"


def test_magnet_not_yet_supported():
    with pytest.raises(FHError) as e:
        SourceResolver().resolve("magnet:?xt=urn:btih:0123")
    assert e.value.code == "unsupported_source"
