"""Shared pytest fixtures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend for CI

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from neuromorphic import setup_simulation  # noqa: E402


@pytest.fixture(autouse=True)
def _seed_numpy():
    """Deterministic RNG for every test."""
    np.random.seed(42)
    yield


@pytest.fixture
def small_params() -> dict:
    """Fast, non-percolating test network."""
    return {
        "NUM_WIRES": 200,
        "T_PULSE": 0.1,
        "T_RELAX": 0.2,
        "TIME_STEP_DT": 1e-2,
    }


@pytest.fixture
def percolating_params() -> dict:
    """Percolating test network, above the percolation threshold."""
    return {
        "NUM_WIRES": 1300,
        "T_PULSE": 0.05,
        "T_RELAX": 0.05,
        "TIME_STEP_DT": 1e-2,
    }


@pytest.fixture
def sim_small(small_params):
    return setup_simulation(parms=small_params)


@pytest.fixture
def sim_percolating(percolating_params):
    return setup_simulation(parms=percolating_params)