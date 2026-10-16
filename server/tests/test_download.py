import hashlib

import httpx
import pytest
from framehound.download import download
from framehound.errors import FHError
from framehound.safe_http import SafeClient

BODY = bytes(range(256)) * 4096  # 1 MiB


def client_for(handler):
    return SafeClient(transport=httpx.MockTransport(handler), resolve=lambda h, p: ["93.184.216.34"])


def test_resume_with_range(tmp_path):
    dest = tmp_path / "source.mp4"
    (tmp_path / "source.mp4.part").write_bytes(BODY[:1000])
    seen = []

    def handler(req):
        seen.append(req.headers.get("range"))
        start = int(req.headers["range"].split("=")[1].rstrip("-"))
        return httpx.Response(206, content=BODY[start:])

    size = download(client_for(handler), "http://x.test/a.mp4", dest, len(BODY),
                    hashlib.md5(BODY).hexdigest(), lambda *a: None, lambda: None)
    assert seen == ["bytes=1000-"] and size == len(BODY) and dest.read_bytes() == BODY


def test_server_ignoring_range_restarts(tmp_path):
    dest = tmp_path / "source.mp4"
    (tmp_path / "source.mp4.part").write_bytes(b"garbage")
    size = download(client_for(lambda req: httpx.Response(200, content=BODY)),
                    "http://x.test/a.mp4", dest, None, None, lambda *a: None, lambda: None)
    assert size == len(BODY) and dest.read_bytes() == BODY


def test_md5_mismatch(tmp_path):
    with pytest.raises(FHError) as e:
        download(client_for(lambda req: httpx.Response(200, content=BODY)), "http://x.test/a",
                 tmp_path / "s.mp4", None, "0" * 32, lambda *a: None, lambda: None)
    assert e.value.code == "download_failed"
    assert not (tmp_path / "s.mp4.part").exists()
