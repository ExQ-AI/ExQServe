from __future__ import annotations

import socket

import pytest

from exqserve.runtime import exllamav3 as runtime_module


def _resolved(ip: str) -> list[tuple[int, int, int, str, tuple[str, int]]]:
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, 443))]


def test_remote_image_url_accepts_only_globally_routable_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _resolved("93.184.216.34"),
    )
    source = "https://example.com/image.png"
    assert runtime_module._validate_remote_image_url(source) == source


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "0.0.0.0"])
def test_remote_image_url_rejects_non_global_destinations(
    monkeypatch: pytest.MonkeyPatch,
    ip: str,
) -> None:
    monkeypatch.setattr(
        runtime_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _resolved(ip),
    )
    with pytest.raises(ValueError, match="globally routable"):
        runtime_module._validate_remote_image_url("http://example.invalid/image.png")


def test_remote_image_url_rejects_mixed_public_private_dns_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _resolved("93.184.216.34") + _resolved("10.0.0.5"),
    )
    with pytest.raises(ValueError, match="globally routable"):
        runtime_module._validate_remote_image_url("https://example.invalid/image.png")


def test_remote_image_url_rejects_embedded_credentials_before_fetch() -> None:
    with pytest.raises(ValueError, match="credentials"):
        runtime_module._validate_remote_image_url("https://user:pass@example.com/image.png")


def test_remote_image_resolution_returns_only_the_validated_connection_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runtime_module.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: _resolved("93.184.216.34"),
    )
    resolved = runtime_module._resolve_remote_image_url("https://example.com/image.png?x=1")
    assert resolved.hostname == "example.com"
    assert resolved.port == 443
    assert resolved.addresses == ("93.184.216.34",)
    assert resolved.request_target == "/image.png?x=1"


def test_pinned_http_connection_connects_to_validated_ip_not_hostname(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = runtime_module._PinnedHTTPConnection(
        "example.com",
        80,
        pinned_address="93.184.216.34",
        timeout=15,
    )
    seen: list[object] = []
    fake_socket = object()

    def create_connection(address: object, timeout: object) -> object:
        seen.append((address, timeout))
        return fake_socket

    monkeypatch.setattr(runtime_module.socket, "create_connection", create_connection)
    connection.connect()
    assert seen == [(('93.184.216.34', 80), 15)]
    assert connection.sock is fake_socket


def test_pinned_https_connection_uses_original_hostname_for_tls_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []
    raw_socket = object()
    tls_socket = object()

    def create_connection(address: object, timeout: object) -> object:
        seen.append(("connect", address, timeout))
        return raw_socket

    class _Context:
        def wrap_socket(self, sock: object, *, server_hostname: str) -> object:
            seen.append(("tls", sock, server_hostname))
            return tls_socket

    context = _Context()
    connection = runtime_module._PinnedHTTPSConnection(
        "example.com",
        443,
        pinned_address="93.184.216.34",
        timeout=15,
        context=context,  # type: ignore[arg-type]
    )
    monkeypatch.setattr(runtime_module.socket, "create_connection", create_connection)
    connection.connect()
    assert seen == [
        ("connect", ("93.184.216.34", 443), 15),
        ("tls", raw_socket, "example.com"),
    ]
    assert connection.sock is tls_socket


def test_remote_image_fetch_revalidates_and_pins_every_redirect_hop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dns_answers = {
        "example.com": "93.184.216.34",
        "cdn.example.com": "142.250.72.14",
    }
    resolved_hosts: list[str] = []

    def getaddrinfo(host: str, *_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        resolved_hosts.append(host)
        return _resolved(dns_answers[host])

    monkeypatch.setattr(runtime_module.socket, "getaddrinfo", getaddrinfo)

    class _Socket:
        def settimeout(self, _timeout: float) -> None:
            return None

    class _Connection:
        sock = _Socket()

        def close(self) -> None:
            return None

    class _Response:
        def __init__(self, status: int, *, location: str | None = None, data: bytes = b"") -> None:
            self.status = status
            self.headers: dict[str, str] = {}
            self._location = location
            self._data = data

        def getheader(self, name: str) -> str | None:
            return self._location if name.lower() == "location" else None

        def read1(self, _size: int) -> bytes:
            data, self._data = self._data, b""
            return data

    opened: list[tuple[str, tuple[str, ...]]] = []
    responses = iter(
        [
            _Response(302, location="https://cdn.example.com/final.png"),
            _Response(200, data=b"image-bytes"),
        ]
    )

    def open_response(resolved, _deadline):  # type: ignore[no-untyped-def]
        opened.append((resolved.hostname, resolved.addresses))
        return _Connection(), next(responses)

    monkeypatch.setattr(runtime_module, "_open_remote_image_response", open_response)
    assert runtime_module._remote_image_bytes("https://example.com/image.png", 1024) == b"image-bytes"
    assert resolved_hosts == ["example.com", "cdn.example.com"]
    assert opened == [
        ("example.com", ("93.184.216.34",)),
        ("cdn.example.com", ("142.250.72.14",)),
    ]


def test_remote_image_slow_drip_obeys_total_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = [0.0]
    monkeypatch.setattr(runtime_module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(runtime_module, "_REMOTE_IMAGE_TOTAL_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(
        runtime_module,
        "_resolve_remote_image_url",
        lambda _source: runtime_module._ResolvedRemoteImageUrl(
            "https://example.com/image.png",
            "https",
            "example.com",
            443,
            ("93.184.216.34",),
            "/image.png",
        ),
    )

    timeouts: list[float] = []

    class _Socket:
        def settimeout(self, timeout: float) -> None:
            timeouts.append(timeout)

    class _Connection:
        sock = _Socket()

        def close(self) -> None:
            return None

    class _Response:
        status = 200

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def read1(self, _size: int) -> bytes:
            now[0] += 0.4
            return b"x"

    monkeypatch.setattr(
        runtime_module,
        "_open_remote_image_response",
        lambda _resolved, _deadline: (_Connection(), _Response()),
    )

    with pytest.raises(TimeoutError, match="total timeout"):
        runtime_module._remote_image_bytes("https://example.com/image.png", 1024)

    assert timeouts == pytest.approx([1.0, 0.6, 0.2])


def test_remote_image_redirect_to_private_dns_is_rejected_before_second_connect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def getaddrinfo(host: str, *_args: object, **_kwargs: object):  # type: ignore[no-untyped-def]
        if host == "example.com":
            return _resolved("93.184.216.34")
        if host == "internal.invalid":
            return _resolved("127.0.0.1")
        raise AssertionError(f"unexpected host {host}")

    monkeypatch.setattr(runtime_module.socket, "getaddrinfo", getaddrinfo)

    class _Connection:
        def close(self) -> None:
            return None

    class _Response:
        status = 302

        def __init__(self) -> None:
            self.headers: dict[str, str] = {}

        def getheader(self, name: str) -> str | None:
            if name.lower() == "location":
                return "http://internal.invalid/private.png"
            return None

    opened: list[str] = []

    def open_response(resolved, _deadline):  # type: ignore[no-untyped-def]
        opened.append(resolved.hostname)
        return _Connection(), _Response()

    monkeypatch.setattr(runtime_module, "_open_remote_image_response", open_response)
    with pytest.raises(ValueError, match="globally routable"):
        runtime_module._remote_image_bytes("https://example.com/image.png", 1024)
    assert opened == ["example.com"]
