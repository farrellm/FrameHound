import httpx
import pytest
from conftest import fixture_json
from fastapi.testclient import TestClient
from framehound import bitmagnet, config, queue
from framehound.errors import FHError


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def respond(payload, status=200):
    return mock_client(lambda req: httpx.Response(status, json=payload))


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("BITMAGNET_URL", "http://bitmagnet:3333/")
    monkeypatch.setenv("MAX_BYTES", str(20 * 1024**3))
    config.reset_settings()


@pytest.fixture
def client():
    from framehound.api import app

    with TestClient(app) as c:
        yield c


def test_groups_releases_under_films(configured):
    seen = []

    def handler(req):
        seen.append(req)
        return httpx.Response(200, json=fixture_json("bitmagnet_search.json"))

    r = bitmagnet.search("night of the living dead", client=mock_client(handler))
    assert str(seen[0].url) == "http://bitmagnet:3333/graphql"
    assert r["total"] == 4
    assert [f["title"] for f in r["films"]] == [
        "Night of the Living Dead", "Night of the Living Dead: Reanimated"]
    notld = r["films"][0]
    assert (notld["year"], notld["runtime_min"]) == (1968, 96)
    # Most seeders first, whatever the relevance order.
    assert [x["seeders"] for x in notld["releases"]] == [30, 8]
    assert notld["releases"][1]["resolution"] == "1080p"
    assert notld["releases"][1]["magnet"].startswith("magnet:?xt=urn:btih:8bc7")
    [pack] = r["unmatched"]
    assert pack["files"] == 124 and pack["too_big"]
    assert not notld["releases"][0]["too_big"]


def test_graphql_error_is_unreachable(configured):
    with pytest.raises(FHError) as e:
        bitmagnet.search("x", client=respond({"errors": [{"message": "boom"}]}))
    assert e.value.code == "bitmagnet_unreachable" and "boom" in e.value.detail


def test_connection_error_is_unreachable(configured):
    def handler(req):
        raise httpx.ConnectError("refused", request=req)

    with pytest.raises(FHError) as e:
        bitmagnet.search("x", client=mock_client(handler))
    assert e.value.code == "bitmagnet_unreachable" and e.value.status == 502


def test_endpoint_off_without_url(client):
    r = client.get("/torrents/search", params={"q": "detour"})
    assert r.status_code == 404 and r.json()["code"] == "not_configured"
    assert client.get("/health").json()["bitmagnet"]["state"] == "not_configured"


def test_endpoint_searches(configured, client, monkeypatch):
    monkeypatch.setattr(bitmagnet, "search", lambda q: {"films": [], "unmatched": [],
                                                        "total": 0, "q": q})
    assert client.get("/health").json()["bitmagnet"]["state"] == "ok"
    assert client.get("/torrents/search", params={"q": " detour "}).json()["q"] == "detour"
    assert client.get("/torrents/search", params={"q": "d"}).status_code == 422


def test_add_magnet_with_title(client, monkeypatch):
    monkeypatch.setattr(queue, "enqueue", lambda kind, job_id: None)
    magnet = fixture_json("bitmagnet_search.json")["data"]["torrentContent"]["search"][
        "items"][0]["torrent"]["magnetUri"]
    r = client.post("/movies", json={"source": magnet,
                                     "title": "Night of the Living Dead (1968)"})
    assert r.status_code == 201
    movie = client.get(f"/movies/{r.json()['movie_id']}").json()
    assert movie["title"] == "Night of the Living Dead (1968)"
