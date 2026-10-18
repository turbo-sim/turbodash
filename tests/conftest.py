"""Shared prescribed inputs for production loss-model integration tests."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def blade_row_case():
    """Return fresh row-frame inputs so each test can vary geometry safely."""
    path = Path(__file__).parent / "fixtures/blade_row_case.json"
    return json.loads(path.read_text(encoding="utf-8"))
