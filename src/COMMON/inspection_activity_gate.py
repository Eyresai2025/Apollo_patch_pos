from __future__ import annotations

"""Mutual exclusion between production capture/AI and PostgreSQL work.

Production work (camera capture, FFC, local image save, AI and visualizations)
has priority. Database jobs wait until production is idle. A new production
cycle waits only when a small metadata transaction is already in progress.
Normal live inspection stores metadata and file paths only; image binaries are
not uploaded to PostgreSQL.
"""

import threading
import time
from contextlib import contextmanager
from typing import Iterator, Optional

_condition = threading.Condition(threading.RLock())
_production_count = 0
_database_count = 0
_production_reason = ""


def begin_production(reason: str = "CAPTURE_AI") -> None:
    global _production_count, _production_reason
    with _condition:
        # Do not start camera acquisition while a PostgreSQL transaction that
        # already obtained the DB slot is still running.
        while _database_count > 0:
            _condition.wait(timeout=0.05)
        _production_count += 1
        _production_reason = str(reason or "CAPTURE_AI")
        _condition.notify_all()


def end_production() -> None:
    global _production_count, _production_reason
    with _condition:
        _production_count = max(0, _production_count - 1)
        if _production_count == 0:
            _production_reason = ""
        _condition.notify_all()


def production_is_active() -> bool:
    with _condition:
        return _production_count > 0


def production_reason() -> str:
    with _condition:
        return _production_reason


def wait_until_production_idle(timeout: Optional[float] = None) -> bool:
    deadline = None if timeout is None else time.monotonic() + max(float(timeout), 0.0)
    with _condition:
        while _production_count > 0:
            if deadline is None:
                _condition.wait(timeout=0.1)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            _condition.wait(timeout=min(0.1, remaining))
        return True


@contextmanager
def database_activity(timeout: Optional[float] = None) -> Iterator[bool]:
    """Acquire a database slot only while production is idle.

    Yields True when acquired. With a finite timeout, yields False when the
    slot could not be acquired in time.
    """
    global _database_count
    deadline = None if timeout is None else time.monotonic() + max(float(timeout), 0.0)
    acquired = False
    with _condition:
        while _production_count > 0:
            if deadline is None:
                _condition.wait(timeout=0.1)
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            _condition.wait(timeout=min(0.1, remaining))
        if _production_count == 0:
            _database_count += 1
            acquired = True
    try:
        yield acquired
    finally:
        if acquired:
            with _condition:
                _database_count = max(0, _database_count - 1)
                _condition.notify_all()
