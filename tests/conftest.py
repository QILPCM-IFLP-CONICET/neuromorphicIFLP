"""Shared pytest fixtures."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sin display, para CI

import pytest

from neuromorphic import setup_simulation


@pytest.fixture
def small_params() -> dict:
    """Params for a fast, non-percolating test network."""
    return {
        "NUM_WIRES": 200,
        "T_PULSE": 0.1,
        "T_RELAX": 0.2,
        "TIME_STEP_DT": 1e-2,
    }


@pytest.fixture
def percolating_params() -> dict:
    """Params for a percolating test network (above percolation threshold)."""
    return {
        "NUM_WIRES": 1200,
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
