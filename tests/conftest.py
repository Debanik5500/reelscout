"""Pytest configuration."""

import pytest
import tempfile
import shutil
from pathlib import Path

from reelscout.storage import get_phone_storage

# Swap the server's Redis connection for an in-memory fakeredis so API tests
# run without a local Redis server. Must happen before any endpoint is called.
try:
    import fakeredis
    from rq import Queue
    from server import main as _server_main

    _server_main.redis_conn = fakeredis.FakeRedis()
    _server_main.task_queue = Queue("reels", connection=_server_main.redis_conn)
    _server_main.multipost_queue = Queue("multipost", connection=_server_main.redis_conn)
except Exception:  # pragma: no cover - fakeredis optional, real Redis may be used
    pass


@pytest.fixture(scope="session", autouse=True)
def cleanup_storage():
    """Clean up test storage after all tests."""
    yield
    # Cleanup after all tests
    storage = get_phone_storage()
    for job_id in list(storage.metadata.keys()):
        if job_id.startswith("test_") or job_id.startswith("test"):
            storage.cleanup_job(job_id)


@pytest.fixture(autouse=True)
def reset_storage():
    """Reset storage before each test."""
    storage = get_phone_storage()
    # Clean up any test jobs
    for job_id in list(storage.metadata.keys()):
        if job_id.startswith("test_"):
            storage.cleanup_job(job_id)
    yield