"""Tests específicos del evolver ``stochastic2`` (modelo de Lamas et al., 2026)."""

import numpy as np
import pytest

from neuromorphic.fisica import EVOLVE_MODELS

stochastic2 = EVOLVE_MODELS["stochastic2"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _set_state(sim, g_value: float) -> None:
    """Fija todos los memristores (array y espejo del grafo) en ``g_value``."""
    circuit = sim["circuit"]
    circuit["memristor_g"][:] = g_value
    for u, v in circuit["mem_edge_keys"]:
        sim["graph"].edges[u, v]["conductance"] = g_value


def _voltages_with_vmem(sim, v_mem: float) -> np.ndarray:
    """Vector nodal con caída ``v_mem`` sobre cada memristor.

    Cada nodo del grafo pertenece a exactamente un memristor (topología
    por duplicación de nodos), así que basta con fijar un extremo en
    ``v_mem`` y el otro en 0.
    """
    circuit = sim["circuit"]
    V = np.zeros(circuit["N"])
    V[circuit["mem_u_idx"]] = v_mem
    return V


def _assert_binomial(n_switched: int, n: int, p: float, n_sigma: float = 5.0) -> None:
    sigma = np.sqrt(n * p * (1 - p))
    assert abs(n_switched - n * p) < n_sigma * sigma, (n_switched, n * p, sigma)


def test_stochastic2_is_registered():
    assert "stochastic2" in EVOLVE_MODELS


def test_each_node_belongs_to_one_memristor(sim_percolating):
    """Precondición de ``_voltages_with_vmem``."""
    c = sim_percolating["circuit"]
    idx = np.concatenate([c["mem_u_idx"], c["mem_v_idx"]])
    assert len(np.unique(idx)) == len(idx)


# ---------------------------------------------------------------------------
# SET
# ---------------------------------------------------------------------------
def test_no_set_below_threshold(sim_percolating):
    p = sim_percolating["parameters"]
    _set_state(sim_percolating, p["G_OFF"])
    V = _voltages_with_vmem(sim_percolating, 0.5 * p["V_THRESHOLD"])
    p["P0_SET"] = 1e12  # incluso con tasa enorme, sin sobretensión no hay SET

    stochastic2(sim_percolating, V)

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_OFF"])


def test_set_rate_matches_formula(sim_percolating):
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 1.0
    overdrive = 1.0
    _set_state(sim_percolating, p["G_OFF"])
    V = _voltages_with_vmem(sim_percolating, p["V_THRESHOLD"] + overdrive)

    stochastic2(sim_percolating, V)

    mem_g = sim_percolating["circuit"]["memristor_g"]
    expected = p["TIME_STEP_DT"] * p["P0_SET"] * (1 - np.exp(-p["ALPHA_SET"] * overdrive))
    _assert_binomial(int((mem_g == p["G_ON"]).sum()), mem_g.size, expected)


def test_set_probability_scales_with_dt(sim_percolating):
    """Con Δt diez veces menor, la tasa de SET por paso cae diez veces."""
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 0.1
    p["P0_SET"] = 5.0
    _set_state(sim_percolating, p["G_OFF"])
    V = _voltages_with_vmem(sim_percolating, 1e3)  # saturación: 1 - exp(-alpha * overdrive) ~ 1

    stochastic2(sim_percolating, V)

    mem_g = sim_percolating["circuit"]["memristor_g"]
    _assert_binomial(int((mem_g == p["G_ON"]).sum()), mem_g.size, 0.5)


# ---------------------------------------------------------------------------
# RESET
# ---------------------------------------------------------------------------
def test_no_reset_without_current(sim_percolating):
    p = sim_percolating["parameters"]
    p["P_DECAY"] = 1e30
    _set_state(sim_percolating, p["G_ON"])

    stochastic2(sim_percolating, np.zeros(sim_percolating["circuit"]["N"]))

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_ON"])


def test_reset_rate_matches_formula(sim_percolating):
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 1.0
    v_mem = 1.0
    i_mem = p["G_ON"] * v_mem
    target = 0.2
    p["P_DECAY"] = target / (p["TIME_STEP_DT"] * i_mem**2)
    _set_state(sim_percolating, p["G_ON"])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, v_mem))

    mem_g = sim_percolating["circuit"]["memristor_g"]
    _assert_binomial(int((mem_g == p["G_OFF"]).sum()), mem_g.size, target)


def test_reset_probability_is_capped_at_one(sim_percolating):
    p = sim_percolating["parameters"]
    p["P_DECAY"] = 1e30
    _set_state(sim_percolating, p["G_ON"])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_OFF"])


# ---------------------------------------------------------------------------
# Consistencia
# ---------------------------------------------------------------------------
def test_no_set_and_reset_in_same_step(sim_percolating):
    """Una juntura que hace SET no puede hacer RESET en el mismo paso."""
    p = sim_percolating["parameters"]
    p["P0_SET"] = 1e30
    p["P_DECAY"] = 1e30
    _set_state(sim_percolating, p["G_OFF"])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    assert np.all(sim_percolating["circuit"]["memristor_g"] == p["G_ON"])


@pytest.mark.parametrize("initial", ["G_OFF", "G_ON"])
def test_graph_mirrors_memristor_array(sim_percolating, initial):
    p = sim_percolating["parameters"]
    p["TIME_STEP_DT"] = 1.0
    p["P_DECAY"] = 0.3 / p["G_ON"] ** 2
    _set_state(sim_percolating, p[initial])

    stochastic2(sim_percolating, _voltages_with_vmem(sim_percolating, 1.0))

    circuit = sim_percolating["circuit"]
    G = sim_percolating["graph"]
    for j, (u, v) in enumerate(circuit["mem_edge_keys"]):
        assert G.edges[u, v]["conductance"] == circuit["memristor_g"][j]


def test_full_run_with_stochastic2(sim_percolating):
    from neuromorphic.dinamica import run_simulation_dynamic_pulse

    sim_percolating["parameters"]["EVOLVER"] = "stochastic2"
    t, g, active = run_simulation_dynamic_pulse(sim_percolating)

    assert len(t) == len(g) == len(active) > 0
    assert np.all(np.isfinite(g))
