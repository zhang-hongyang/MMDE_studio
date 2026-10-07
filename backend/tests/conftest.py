"""Contract tests against the real read-only datasets (no mocks)."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import create_app  # noqa: E402


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(create_app())


def read_u32(buf: bytes, off: int) -> int:
    return struct.unpack_from("<I", buf, off)[0]
