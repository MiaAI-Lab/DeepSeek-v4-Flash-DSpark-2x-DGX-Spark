#!/usr/bin/env python3
"""Correlate a soak's streamed completion IDs with LiteLLM spend rows."""

from __future__ import annotations

import re
import subprocess
import time
from collections.abc import Callable, Sequence


RESPONSE_ID = re.compile(r"chatcmpl-[a-f0-9]{16}")
POSTGRES_COMMAND = (
    "docker", "exec", "dspark-private-litellm-postgres-1",
    "psql", "-X", "-A", "-t", "-U", "litellm_smoke", "-d", "litellm_smoke",
    "-c",
)


def validate_response_ids(response_ids: Sequence[str]) -> None:
    if len(set(response_ids)) != len(response_ids):
        raise RuntimeError("soak origin response IDs are duplicated")
    if not all(RESPONSE_ID.fullmatch(item) for item in response_ids):
        raise RuntimeError("soak origin response ID format is invalid")


def spend_count(response_ids: Sequence[str]) -> int:
    if not response_ids:
        return 0
    validate_response_ids(response_ids)
    ids = ",".join("'" + item + "'" for item in response_ids)
    query = (
        'SELECT count(*) FROM "LiteLLM_SpendLogs" AS t '
        f"WHERE t.request_id IN ({ids});"
    )
    output = subprocess.check_output([*POSTGRES_COMMAND, query], text=True)
    return int(output.strip())


def wait_for_spend(
    response_ids: Sequence[str],
    *,
    count_fn: Callable[[Sequence[str]], int] = spend_count,
    timeout: float = 120.0,
    interval: float = 1.0,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Return the exact count, or the last partial count after a bounded wait.

    A partial count makes the acceptance result false while still allowing the
    caller to persist the completed 30-minute soak evidence. Invalid IDs and
    duplicate spend rows remain immediate hard failures.
    """
    validate_response_ids(response_ids)
    expected = len(response_ids)
    deadline = monotonic() + timeout
    current = 0
    while monotonic() < deadline:
        current = count_fn(response_ids)
        if current == expected:
            return current
        if current > expected:
            raise RuntimeError("soak response IDs produced duplicate spend rows")
        sleep(interval)
    return current
