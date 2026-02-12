from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse


class URLValidationError(Exception):
    """Raised when a URL fails security validation."""


def _is_ip_blocked(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if the IP is private, reserved, loopback, or link-local.

    Also blocks IPv4-mapped IPv6 addresses (e.g. ``::ffff:127.0.0.1``)
    to prevent bypass via IPv6 encoding of private IPv4 addresses.
    """
    # Unwrap IPv4-mapped IPv6 addresses before checking
    if (
        isinstance(ip, ipaddress.IPv6Address)
        and ip.ipv4_mapped is not None
    ):
        ip = ip.ipv4_mapped

    return (
        ip.is_private
        or ip.is_reserved
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_unspecified
    )


def validate_url(
    url: str,
    allowed_domains: list[str] | None = None,
    blocked_domains: list[str] | None = None,
) -> str:
    """Validate a URL for safety. Returns the normalized URL.

    Checks:
    - Must be http or https scheme
    - Must not contain credentials (userinfo) in the URL
    - Must not resolve to private/reserved IP addresses (SSRF)
    - Blocks IPv4-mapped IPv6 addresses (``::ffff:127.0.0.1``)
    - Must not be on the blocked_domains list
    - If allowed_domains is set, must be on that list
    """
    parsed = urlparse(url)

    # Scheme check
    if parsed.scheme not in ("http", "https"):
        raise URLValidationError(
            f"Invalid scheme: {parsed.scheme!r}."
            " Only http/https allowed."
        )

    hostname = parsed.hostname
    if not hostname:
        raise URLValidationError("URL has no hostname")

    # Block credentials in URL (often used in SSRF bypass)
    if parsed.username or parsed.password:
        raise URLValidationError(
            "URLs with embedded credentials are not allowed"
        )

    # Block bare IP addresses to reduce SSRF surface
    # (the DNS resolution check below also catches them, but
    #  this provides defence-in-depth for edge cases)
    try:
        literal_ip = ipaddress.ip_address(hostname)
        if _is_ip_blocked(literal_ip):
            raise URLValidationError(
                f"URL points to blocked IP {literal_ip}."
                " This may be an SSRF attempt."
            )
    except ValueError:
        pass  # Not a literal IP -- expected for hostnames

    # Domain allowlist/blocklist
    if blocked_domains:
        for domain in blocked_domains:
            if (
                hostname == domain
                or hostname.endswith(f".{domain}")
            ):
                raise URLValidationError(
                    f"Domain {hostname!r} is blocked"
                )

    if allowed_domains:
        allowed = False
        for domain in allowed_domains:
            if (
                hostname == domain
                or hostname.endswith(f".{domain}")
            ):
                allowed = True
                break
        if not allowed:
            raise URLValidationError(
                f"Domain {hostname!r} is not in the allowed list"
            )

    # DNS resolution + private IP check (SSRF prevention)
    try:
        addrs = socket.getaddrinfo(hostname, None)
    except socket.gaierror as err:
        raise URLValidationError(
            f"Cannot resolve hostname: {hostname!r}"
        ) from err

    if not addrs:
        raise URLValidationError(
            f"Hostname {hostname!r} resolved to no addresses"
        )

    for _, _, _, _, sockaddr in addrs:
        ip = ipaddress.ip_address(sockaddr[0])
        if _is_ip_blocked(ip):
            raise URLValidationError(
                f"URL resolves to blocked IP {ip}."
                " This may be an SSRF attempt."
            )

    return url


def validate_resolved_ip(
    ip_str: str,
) -> None:
    """Validate a resolved IP address is not private/reserved.

    Used as a post-connect check to prevent DNS rebinding and
    redirect-based SSRF attacks.

    Raises:
        URLValidationError: If the IP is in a blocked range.
    """
    ip = ipaddress.ip_address(ip_str)
    if _is_ip_blocked(ip):
        raise URLValidationError(
            f"Connection to blocked IP {ip} denied."
            " Possible SSRF via redirect or DNS rebinding."
        )


class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, rate_per_minute: int = 30, burst: int = 5) -> None:
        import time
        self._rate = rate_per_minute / 60.0
        self._burst = burst
        self._tokens = float(burst)
        self._last_refill = time.time()

    def acquire(self) -> bool:
        """Try to acquire a token. Returns True if allowed, False if rate limited."""
        import time
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False
