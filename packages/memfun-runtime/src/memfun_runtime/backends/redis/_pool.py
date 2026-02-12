from __future__ import annotations

from typing import TYPE_CHECKING

from memfun_core.logging import get_logger

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = get_logger("backend.redis")


async def create_pool(redis_url: str) -> Redis:
    """Create and verify an async Redis connection pool.

    Returns a :class:`redis.asyncio.Redis` instance backed by a connection
    pool.  The connection is validated with a ``PING`` before returning.
    """
    try:
        from redis.asyncio import Redis as AsyncRedis
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "redis extra is required for the Redis backend. "
            "Install with: pip install memfun-runtime[redis]"
        ) from exc

    client: Redis = AsyncRedis.from_url(
        redis_url, decode_responses=False,
    )
    try:
        await client.ping()
    except Exception:
        logger.exception("Failed to connect to Redis at %s", redis_url)
        raise
    logger.info("Connected to Redis at %s", redis_url)
    return client
