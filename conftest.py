"""Pytest configuration and opt-in real data test fixtures."""

import os

import pytest


def pytest_addoption(parser):
    """Add --real-data option to pytest CLI."""
    parser.addoption(
        "--real-data",
        action="store_true",
        default=False,
        help="Run real lunar dataset integration tests (requires LUNAR_REAL_DATA_DIR env var)",
    )


@pytest.fixture
def real_data_dir(request):
    """Fixture supplying path to real dataset directory or skipping cleanly if not provided."""
    if not request.config.getoption("--real-data"):
        pytest.skip("Skipping real data tests (pass --real-data option to run)")
    path = os.environ.get("LUNAR_REAL_DATA_DIR")
    if not path or not os.path.isdir(path):
        pytest.skip(
            "LUNAR_REAL_DATA_DIR environment variable not set or directory missing — skipping real-data test"
        )
    return path
