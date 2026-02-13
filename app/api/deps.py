"""
Dependency injection for API routes.
"""

import time
from typing import AsyncGenerator

from fastapi import Depends

from app.core.config import get_settings
from app.core.logging import logger

# Service singletons
_quality_service = None
_aircraft_service = None
_airline_service = None
_registration_service = None
_review_service = None


# Request counter for stats
# NOTE: These counters are process-local and not suitable for multi-worker deployments.
# In production with multiple workers (e.g., uvicorn --workers N), each worker maintains
# its own independent counters, leading to inaccurate statistics. For accurate
# application-wide statistics, consider using a centralized storage like Redis.
_request_count = 0
_success_count = 0
_error_count = 0
_start_time = time.time()


async def get_request_stats() -> dict:
    """Get current request statistics."""
    uptime = time.time() - _start_time
    rps = _request_count / uptime if uptime > 0 else 0

    return {
        "total_requests": _request_count,
        "successful_requests": _success_count,
        "failed_requests": _error_count,
        "uptime_seconds": uptime,
        "requests_per_second": rps
    }


def increment_request_count(success: bool = True) -> None:
    """Increment request counters."""
    global _request_count, _success_count, _error_count
    _request_count += 1
    if success:
        _success_count += 1
    else:
        _error_count += 1
