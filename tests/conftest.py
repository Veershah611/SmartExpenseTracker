"""
tests/conftest.py
=================
Pytest configuration for the Smart Expense Tracker test suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is always on sys.path for imports.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "online: marks tests that require a running Ollama instance (deselect with '-m \"not online\"')",
    )
