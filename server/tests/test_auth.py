"""FH_TOKEN: a shared bearer token, or the cookie POST /auth sets."""

import pytest
from fastapi.testclient import TestClient
from framehound import config
from test_face_search import ready_movie
from test_pipeline_flow import enqueued  # noqa: F401 (fixture)


@pytest.fixture
def app_client():
    from framehound.api import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def token(monkeypatch):
    monkeypatch.setenv("FH_TOKEN", "s3cret")
    config.reset_settings()
    return "s3cret"


def test_off_by_default(app_client):
    assert app_client.get("/movies").status_code == 200
    assert app_client.get("/health").json()["auth_required"] is False


def test_requests_need_the_token(token, app_client):
    r = app_client.get("/movies")
    assert r.status_code == 401 and r.json()["code"] == "auth_required"
    assert app_client.get("/movies", headers={"Authorization": "Bearer nope"}).status_code == 401
    ok = app_client.get("/movies", headers={"Authorization": f"Bearer {token}"})
    assert ok.status_code == 200
    # Health says only that a token is needed.
    assert app_client.get("/health").json() == {"ok": True, "auth_required": True,
                                                "signed_in": False}


def test_sign_in_sets_a_cookie_for_media(token, app_client, data_dir, enqueued):  # noqa: F811
    ready_movie(data_dir, enqueued)
    assert app_client.get("/media/mv1/proxy.mp4").status_code == 401
    bad = app_client.post("/auth", json={"token": "wrong"})
    assert bad.status_code == 401 and bad.json()["code"] == "bad_token"
    r = app_client.post("/auth", json={"token": token})
    assert r.status_code == 200
    cookie = r.cookies.get("fh_auth")
    assert cookie and token not in cookie
    assert "httponly" in r.headers["set-cookie"].lower()
    assert app_client.get("/media/mv1/proxy.mp4").status_code == 200
    assert app_client.get("/media/mv1/thumbs/0.jpg").status_code == 200
    assert app_client.get("/health").json()["signed_in"] is True
    assert app_client.delete("/auth").status_code == 204
    app_client.cookies.clear()
    assert app_client.get("/movies").status_code == 401


def test_changing_the_token_signs_out(token, app_client, monkeypatch):
    app_client.post("/auth", json={"token": token})
    assert app_client.get("/movies").status_code == 200
    monkeypatch.setenv("FH_TOKEN", "rotated")
    config.reset_settings()
    assert app_client.get("/movies").status_code == 401


def test_secure_cookie_behind_https(token, app_client):
    r = app_client.post("/auth", json={"token": token}, headers={"X-Forwarded-Proto": "https"})
    assert "secure" in r.headers["set-cookie"].lower()
