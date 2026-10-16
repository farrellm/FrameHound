import httpx
import pytest
from framehound.errors import FHError
from framehound.safe_http import SafeClient, is_public, vet


@pytest.mark.parametrize("ip", [
    "127.0.0.1", "10.1.2.3", "172.16.0.1", "192.168.1.1", "169.254.169.254", "100.64.0.1",
    "0.0.0.0", "::1", "fe80::1", "fd00::1", "::ffff:127.0.0.1", "224.0.0.1",
])
def test_private_addresses_blocked(ip):
    assert not is_public(ip)


@pytest.mark.parametrize("ip", ["8.8.8.8", "207.241.224.2", "2606:4700:4700::1111"])
def test_public_addresses_allowed(ip):
    assert is_public(ip)


def test_vet_rewrites_to_ip():
    url, host = vet("https://example.com:8443/a?b=1", lambda h, p: ["93.184.216.34"])
    assert host == "example.com"
    assert str(url) == "https://93.184.216.34:8443/a?b=1"


def test_vet_rejects_any_private_answer():
    # One private A record is enough to refuse (no picking the "good" one).
    with pytest.raises(FHError) as e:
        vet("http://evil.test/", lambda h, p: ["93.184.216.34", "10.0.0.5"])
    assert e.value.code == "blocked_address"


def test_vet_rejects_literal_private_ip_and_bad_scheme():
    with pytest.raises(FHError):
        vet("http://169.254.169.254/latest/meta-data", lambda h, p: [])
    with pytest.raises(FHError) as e:
        vet("file:///etc/passwd", lambda h, p: [])
    assert e.value.code == "unrecognized"


def test_redirect_to_private_ip_blocked():
    dns = {"public.test": ["93.184.216.34"], "internal.test": ["192.168.0.10"]}

    def handler(req: httpx.Request) -> httpx.Response:
        if req.headers["host"] == "public.test":
            return httpx.Response(302, headers={"location": "http://internal.test/secret"})
        return httpx.Response(200, content=b"secret")

    client = SafeClient(transport=httpx.MockTransport(handler), resolve=lambda h, p: dns[h])
    with pytest.raises(FHError) as e, client.stream("http://public.test/movie.mp4"):
        pass
    assert e.value.code == "blocked_address"


def test_dns_rebinding_connects_to_vetted_ip():
    # The resolver answers public first, private second; we must connect to the
    # first (vetted) answer and never re-resolve.
    answers = iter([["93.184.216.34"], ["127.0.0.1"]])
    seen = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.host)
        return httpx.Response(200, content=b"ok")

    client = SafeClient(transport=httpx.MockTransport(handler),
                        resolve=lambda h, p: next(answers))
    with client.stream("http://rebind.test/x") as r:
        r.read()
    assert seen == ["93.184.216.34"]


def test_too_many_redirects():
    def handler(req):
        return httpx.Response(301, headers={"location": "/again"})

    client = SafeClient(transport=httpx.MockTransport(handler),
                        resolve=lambda h, p: ["93.184.216.34"])
    with pytest.raises(FHError, match="redirects too many"), client.stream("http://loop.test/"):
        pass
