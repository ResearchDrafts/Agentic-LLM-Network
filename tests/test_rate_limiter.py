import asyncio
import time


from sandbox.rate_limiter import RateLimiter


async def test_acquire_spaces_requests_no_closer_than_interval():
    # 600/minute -> 0.1s minimum interval, easy to measure reliably in a test
    limiter = RateLimiter({"openai": 600})
    start = time.monotonic()
    for _ in range(4):
        await limiter.acquire("openai")
    elapsed = time.monotonic() - start
    assert elapsed >= 0.3 * 0.9  # 3 intervals of ~0.1s, with slack for scheduling jitter


async def test_unconfigured_provider_returns_immediately():
    limiter = RateLimiter({"openai": 1})  # very tight limit for a configured provider
    start = time.monotonic()
    for _ in range(20):
        await limiter.acquire("local")  # never configured -> no limit
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


async def test_concurrent_acquire_for_same_provider_serializes_correctly():
    limiter = RateLimiter({"openai": 600})
    start = time.monotonic()
    await asyncio.gather(*[limiter.acquire("openai") for _ in range(5)])
    elapsed = time.monotonic() - start
    # 5 calls at 0.1s min interval each -> at least ~0.4s total once serialized
    assert elapsed >= 0.4 * 0.8


async def test_zero_limit_is_treated_as_no_limit_not_division_by_zero():
    limiter = RateLimiter({"openai": 0})
    start = time.monotonic()
    for _ in range(20):
        await limiter.acquire("openai")
    elapsed = time.monotonic() - start
    assert elapsed < 0.5


def test_release_is_a_no_op():
    limiter = RateLimiter({"openai": 100})
    limiter.release("openai")  # must not raise
    limiter.release("never_configured")  # must not raise either
