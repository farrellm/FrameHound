"""SSRF-guarded HTTP: vet every hop's address, then connect to that vetted IP.

The host is resolved once, every address is checked, and the request goes to
the vetted IP with the original ``Host`` header and TLS SNI/hostname, so a DNS
rebind between check and connect can't redirect us. Redirects are followed by
hand (at most 5) and re-validated.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Self
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from .config import settings
from .errors import FHError

MAX_REDIRECTS = 5
USER_AGENT = "FrameHound/0.1 (+self-hosted)"

Resolver = Callable[[str, int], list[str]]

BLOCKED_MESSAGE = "FrameHound can't download from private network addresses. Use a public link."


def system_resolve(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    return list(dict.fromkeys(i[4][0] for i in infos))


def is_public(ip: str) -> bool:
    addr = ipaddress.ip_address(ip.split("%")[0])
    if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped:
        addr = addr.ipv4_mapped
    # is_global excludes loopback, RFC1918, link-local (incl. 169.254.169.254),
    # CGNAT 100.64/10, unique-local, documentation and reserved ranges.
    return addr.is_global and not addr.is_multicast


def vet(url: str, resolve: Resolver) -> tuple[httpx.URL, str]:
    """Return (url rewritten to a vetted IP, original host)."""
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise FHError("unrecognized", "That isn't a web address or magnet link. "
                      "Paste a link that starts with https:// or magnet:.")
    if parts.username or parts.password:
        raise FHError("unrecognized", "Links with a username or password aren't supported.")
    host = parts.hostname
    port = parts.port or (443 if parts.scheme == "https" else 80)
    try:
        ips = [host] if _is_ip(host) else resolve(host, port)
    except (socket.gaierror, UnicodeError) as e:
        raise FHError("unreachable", "FrameHound couldn't find that server. Check the link.",
                      detail=f"DNS lookup for {host} failed: {e}") from e
    if not ips:
        raise FHError("unreachable", "FrameHound couldn't find that server. Check the link.")
    if not settings().allow_private_urls:
        bad = [ip for ip in ips if not is_public(ip)]
        if bad:
            raise FHError("blocked_address", BLOCKED_MESSAGE, detail=f"{host} resolves to {bad[0]}")
    ip = ips[0]
    netloc_ip = f"[{ip}]" if ":" in ip else ip
    if parts.port:
        netloc_ip += f":{parts.port}"
    return httpx.URL(urlunsplit((parts.scheme, netloc_ip, parts.path or "/", parts.query, ""))), host


def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


class SafeClient:
    def __init__(self, transport: httpx.BaseTransport | None = None,
                 resolve: Resolver = system_resolve, timeout: float = 30.0):
        self.resolve = resolve
        self.client = httpx.Client(transport=transport, timeout=httpx.Timeout(timeout, read=60.0),
                                   follow_redirects=False)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @contextmanager
    def stream(self, url: str, headers: dict[str, str] | None = None
               ) -> Iterator[httpx.Response]:
        """GET ``url`` with vetted redirects. ``response.url_final`` is the last hop's URL."""
        current = url
        for _ in range(MAX_REDIRECTS + 1):
            target, host = vet(current, self.resolve)
            port = urlsplit(current).port
            hdrs = {"User-Agent": USER_AGENT, **(headers or {}),
                    "Host": host if not port else f"{host}:{port}"}
            req = self.client.build_request("GET", target, headers=hdrs,
                                            extensions={"sni_hostname": host})
            try:
                resp = self.client.send(req, stream=True)
            except httpx.HTTPError as e:
                raise FHError("unreachable", "FrameHound couldn't reach that server. "
                              "Check the link and try again.", detail=str(e)) from e
            if resp.is_redirect:
                location = resp.headers.get("location", "")
                resp.close()
                current = urljoin(current, location)
                continue
            resp.url_final = current  # type: ignore[attr-defined]
            try:
                yield resp
            finally:
                resp.close()
            return
        raise FHError("unreachable", "That link redirects too many times.",
                      detail=f"more than {MAX_REDIRECTS} redirects")

    def get_json(self, url: str) -> object:
        with self.stream(url, {"Accept": "application/json"}) as r:
            r.read()
            if r.status_code >= 400:
                raise FHError("unreachable", "FrameHound couldn't reach that server. "
                              "Check the link and try again.", detail=f"HTTP {r.status_code} for {url}")
            return r.json()
