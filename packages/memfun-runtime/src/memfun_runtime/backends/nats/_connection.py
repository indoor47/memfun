from __future__ import annotations

from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from nats.aio.client import Client as NATSClient
    from nats.js import JetStreamContext

logger = get_logger("backend.nats")


async def connect(
    nats_url: str,
    creds_file: str | None = None,
) -> tuple[NATSClient, JetStreamContext]:
    """Connect to NATS and return the client with a JetStream context.

    Validates the connection before returning.
    """
    try:
        import nats
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "nats extra is required for the NATS backend. "
            "Install with: pip install memfun-runtime[nats]"
        ) from exc

    connect_kwargs: dict = {"servers": [nats_url]}
    if creds_file:
        connect_kwargs["user_credentials"] = creds_file

    try:
        nc: NATSClient = await nats.connect(**connect_kwargs)
    except Exception:
        logger.exception("Failed to connect to NATS at %s", nats_url)
        raise

    js = nc.jetstream()
    logger.info("Connected to NATS at %s (JetStream enabled)", nats_url)
    return nc, js


async def ensure_kv_bucket(
    js: JetStreamContext,
    bucket: str,
    *,
    ttl: int = 0,
    max_value_size: int = -1,
) -> object:
    """Create or bind to a NATS KV bucket.

    Returns a ``KeyValue`` handle.
    """
    try:
        kv = await js.key_value(bucket)
    except Exception:
        from nats.js.api import KeyValueConfig

        cfg = KeyValueConfig(bucket=bucket, ttl=ttl, max_value_size=max_value_size)
        kv = await js.create_key_value(config=cfg)
    return kv
