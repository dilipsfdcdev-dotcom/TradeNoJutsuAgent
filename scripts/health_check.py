#!/usr/bin/env python3
"""Health check script for the trading agent system."""

import sys
import asyncio
import time
import argparse
import json
import os
from datetime import datetime, timezone

import httpx
import redis.asyncio as aioredis


async def check_postgres(url: str) -> tuple[bool, str, float]:
    """Check PostgreSQL connectivity."""
    start = time.time()
    try:
        import asyncpg

        conn = await asyncpg.connect(url)
        await conn.execute("SELECT 1")
        await conn.close()
        latency = time.time() - start
        return True, "Connected", latency
    except Exception as e:
        return False, str(e), time.time() - start


async def check_redis(url: str) -> tuple[bool, str, float]:
    """Check Redis connectivity."""
    start = time.time()
    try:
        r = aioredis.from_url(url)
        await r.ping()
        await r.aclose()
        return True, "Connected", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


async def check_http(name: str, url: str) -> tuple[bool, str, float]:
    """Check HTTP endpoint reachability."""
    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(url)
            return resp.status_code < 500, f"HTTP {resp.status_code}", time.time() - start
    except Exception as e:
        return False, str(e), time.time() - start


async def main() -> None:
    parser = argparse.ArgumentParser(description="Health check for trading agent")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    db_url = os.getenv("DATABASE_URL", "postgresql://agent:password@localhost:5432/trading_agent")
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    api_port = os.getenv("API_PORT", "8000")
    dash_port = os.getenv("DASHBOARD_PORT", "3000")

    checks: dict[str, tuple[bool, str, float]] = {}
    checks["postgresql"] = await check_postgres(db_url)
    checks["redis"] = await check_redis(redis_url)
    checks["api_server"] = await check_http("API", f"http://localhost:{api_port}/api/metrics")
    checks["dashboard"] = await check_http("Dashboard", f"http://localhost:{dash_port}")

    all_ok = all(v[0] for v in checks.values())

    if args.json:
        result: dict = {
            k: {"ok": v[0], "message": v[1], "latency_ms": round(v[2] * 1000, 1)}
            for k, v in checks.items()
        }
        result["overall"] = "healthy" if all_ok else "unhealthy"
        result["timestamp"] = datetime.now(timezone.utc).isoformat()
        print(json.dumps(result, indent=2))
    else:
        print(f"\n{'=' * 50}")
        print("  Trading Agent Health Check")
        print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'=' * 50}\n")
        for name, (ok, msg, latency) in checks.items():
            status = "\u2713" if ok else "\u2717"
            color = "\033[92m" if ok else "\033[91m"
            reset = "\033[0m"
            print(f"  {color}{status}{reset} {name:<20} {msg:<30} ({latency * 1000:.0f}ms)")
        print(f"\n  Overall: {'HEALTHY' if all_ok else 'UNHEALTHY'}\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
