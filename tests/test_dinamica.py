"""Tests for neuromorphic.simulador."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from neuromorphic.dinamica import run_simulation_dynamic_pulse


# ---------------------------------------------------------------------------
# Happy path: use the session-cached run
# ---------------------------------------------------------------------------
def test_run_returns_three_histories(_pulse_run):
    _, t, g, act = _pulse_run
    assert len(t) == len(g) == len(act)
    assert len(t) > 0


def test_history_length_matches_total_steps(_pulse_run, percolating_params):
    p = percolating_params
    # Debe coincidir con lo que hace simulador.py:
    #   total_steps = int((T_PULSE + T_RELAX) / TIME_STEP_DT)
    expected = int((p["T_PULSE"] + p["T_RELAX"]) / p["TIME_STEP_DT"])
    _, t, _, _ = _pulse_run
    assert len(t) == expected


def test_time_is_monotonic(_pulse_run):
    _, t, _, _ = _pulse_run
    for prev, nxt in pairwise((t, t[1:])):
        assert nxt > prev


def test_active_count_is_non_negative(_pulse_run):
    _, _, _, act = _pulse_run
    assert all(a >= 0 for a in act)


def test_all_histories_are_finite(_pulse_run):
    _, t, g, act = _pulse_run
    assert np.all(np.isfinite(t))
    assert np.all(np.isfinite(g))
    assert np.all(np.isfinite(act))


# ---------------------------------------------------------------------------
# Error path: build a fresh small network and verify the rejection
# ---------------------------------------------------------------------------
def test_non_percolating_raises(sim_small):
    """A network below threshold must be rejected explicitly."""
    with pytest.raises(RuntimeError):
        run_simulation_dynamic_pulse(sim_small)
