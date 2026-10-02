"""Registro de modelos de evolución de memristores y modelos binarios incluidos.

Un modelo de evolución consta de:

- ``update(simulation, V_solved)``: avanza un paso temporal. Debe mantener
  coherentes ``circuit["memristor_g"]``, el espejo
  ``graph.edges[u, v]["conductance"]`` y la máscara
  ``circuit["memristor_active"]``.
- ``init(simulation)`` (opcional): fija el estado inicial de los
  memristores y guarda lo que el modelo necesite precalcular en
  ``simulation["evolver_state"]``. Por defecto, todos en ``G_OFF``.
- ``parameters`` (opcional): parámetros propios del modelo con sus valores
  por defecto. Se aceptan en ``parms=`` sin advertencias y se completan en
  ``simulation["parameters"]`` al inicializar el modelo.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np

EvolverFn = Callable[[dict[str, Any], np.ndarray], Any]
InitFn = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class EvolverSpec:
    """Especificación completa de un modelo de evolución registrado."""

    update: EvolverFn
    init: InitFn
    parameters: Mapping[str, Any] = field(default_factory=dict)


EVOLVER_SPECS: dict[str, EvolverSpec] = {}
"""Nombre -> :class:`EvolverSpec` completa."""

_F = TypeVar("_F", bound=EvolverFn)


def set_all_memristors(simulation: dict[str, Any], g_value: float) -> None:
    """Fija todos los memristores en ``g_value`` (array, grafo y máscara)."""
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]
    circuit["memristor_g"][:] = g_value
    for u, v in circuit["mem_edge_keys"]:
        G.edges[u, v]["conductance"] = g_value
    circuit["memristor_active"] = circuit["memristor_g"] == p["G_ON"]


def init_all_off(simulation: dict[str, Any]) -> None:
    """Inicialización por defecto: todos los memristores en ``G_OFF``."""
    set_all_memristors(simulation, simulation["parameters"]["G_OFF"])


def register_memristor_evol_model(
    name: str,
    *,
    init: InitFn | None = None,
    parameters: Mapping[str, Any] | None = None,
) -> Callable[[_F], _F]:
    """Decorador que registra un modelo de evolución bajo ``name``.

    Parameters
    ----------
    name : str
        Nombre con el que se selecciona el modelo (``EVOLVER``).
    init : callable, optional
        ``init(simulation)``. Si se omite, se usa :func:`init_all_off`.
    parameters : mapping, optional
        Parámetros propios del modelo y sus valores por defecto.

    Registrar un nombre ya existente lo sobrescribe sin aviso.
    """

    def _register(fn: _F) -> _F:
        EVOLVER_SPECS[name] = EvolverSpec(
            update=fn,
            init=init if init is not None else init_all_off,
            parameters=dict(parameters or {}),
        )
        return fn

    return _register


def declared_evolver_parameters() -> frozenset[str]:
    """Claves de parámetros declaradas por algún modelo registrado."""
    return frozenset(k for spec in EVOLVER_SPECS.values() for k in spec.parameters)


def initialize_evolver(simulation: dict[str, Any]) -> EvolverSpec:
    """Prepara el modelo seleccionado en ``parameters["EVOLVER"]``.

    Completa en ``simulation["parameters"]`` los parámetros del modelo que
    no estén definidos, recrea ``simulation["evolver_state"]`` vacío y
    llama a ``init``. Es idempotente: llamarla de nuevo reinicia el estado
    de los memristores.

    Raises
    ------
    KeyError
        Si ``EVOLVER`` no corresponde a ningún modelo registrado.
    """
    p = simulation["parameters"]
    name = p["EVOLVER"]
    if name not in EVOLVER_SPECS:
        raise KeyError(
            f"Modelo de evolución desconocido: {name!r}. Registrados: {sorted(EVOLVER_SPECS)}"
        )
    spec = EVOLVER_SPECS[name]
    for key, default in spec.parameters.items():
        p.setdefault(key, default)
    simulation["evolver_state"] = {}
    spec.init(simulation)
    return spec


# ==============================================================================
# ACTUALIZACIÓN ESTOCÁSTICA DE MEMRISTORES
# ==============================================================================
@register_memristor_evol_model("stochastic1")
def stochastic_updater1(simulation: dict[str, Any], V_solved):
    """Actualiza estocásticamente el estado de los memristores.

    Vectorizado con NumPy sobre los arrays precalculados en
    ``simulation["circuit"]``.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto.

    Returns
    -------
    networkx.Graph
        El mismo objeto ``simulation["graph"]`` mutado in-place.

    Notes
    -----
    El orden de consumo del RNG difiere de la versión secuencial previa:
    con la misma semilla, los memristores que conmutan pueden no ser los
    mismos que antes. La distribución estadística del proceso es idéntica.

    See Also
    --------
    build_admittance_matrix : construye el sistema que produce ``V_solved``.
    """
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]

    mem_g = circuit["memristor_g"]
    mem_u = circuit["mem_u_idx"]
    mem_v = circuit["mem_v_idx"]
    mem_edges = circuit["mem_edge_keys"]

    V_th = p["V_THRESHOLD"]
    G_OFF = p["G_OFF"]
    G_ON = p["G_ON"]

    # Snapshot del estado ANTES de tomar decisiones: así una arista que
    # hace SET en este paso no puede hacer RESET en el mismo paso.
    is_off = mem_g == G_OFF
    is_on = mem_g == G_ON

    v_mem = np.abs(V_solved[mem_u] - V_solved[mem_v])

    # --- SET (OFF -> ON) ---
    set_candidates = is_off & (v_mem > V_th)
    n_set = int(set_candidates.sum())
    if n_set > 0:
        v_cand = v_mem[set_candidates]
        p_set = p["P0_SET"] * np.exp(p["ALPHA_SET"] * (v_cand - V_th))
        p_set = np.clip(p_set, 0.0, 1.0)
        r = np.random.random(n_set)
        cand_local = np.flatnonzero(r < p_set)
        cand_global = np.flatnonzero(set_candidates)[cand_local]

        mem_g[cand_global] = G_ON
        for j in cand_global:
            u, v = mem_edges[j]
            G.edges[u, v]["conductance"] = G_ON

    # --- RESET (ON -> OFF) ---
    reset_candidates = is_on
    n_reset = int(reset_candidates.sum())
    if n_reset > 0:
        v_cand = v_mem[reset_candidates]
        estabilidad = np.exp(-v_cand / V_th)
        p_decay = p["P_DECAY"] * estabilidad
        r = np.random.random(n_reset)
        cand_local = np.flatnonzero(r < p_decay)
        cand_global = np.flatnonzero(reset_candidates)[cand_local]

        mem_g[cand_global] = G_OFF
        for j in cand_global:
            u, v = mem_edges[j]
            G.edges[u, v]["conductance"] = G_OFF

    circuit["memristor_active"] = mem_g == G_ON
    return G


@register_memristor_evol_model("stochastic2")
def stochastic_updater2(simulation: dict[str, Any], V_solved):
    r"""Actualiza los memristores según el modelo de Lamas et al. (2026).

    Implementa las probabilidades de conmutación por paso temporal del
    paper:

    .. math::

        P_\mathrm{set} = \Delta t \, P_0 \,
            \max\!\left(0,\; 1 - e^{-\alpha (V_\mathrm{mem} - V_\mathrm{th})}\right)

        P_\mathrm{reset} = \min\!\left(1,\; \Delta t \, P_\mathrm{decay} \, I_\mathrm{mem}^2\right)

    donde :math:`V_\mathrm{mem} = |V_u - V_v|` es la caída de tensión
    sobre la juntura e :math:`I_\mathrm{mem} = G_\mathrm{mem} V_\mathrm{mem}`
    la corriente que la atraviesa.

    A diferencia de ``stochastic1``, las probabilidades escalan con
    ``TIME_STEP_DT``, de modo que el resultado estadístico no depende del
    paso temporal (para :math:`\Delta t` chico). ``P0_SET`` tiene unidades
    de s⁻¹ y ``P_DECAY`` de A⁻² s⁻¹.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto.

    Returns
    -------
    networkx.Graph
        El mismo objeto ``simulation["graph"]`` mutado in-place.

    Notes
    -----
    - Las decisiones se toman sobre el estado al inicio del paso: una
      juntura que hace SET no puede hacer RESET en el mismo paso.
    - :math:`P_\mathrm{set}` se recorta a 1 por seguridad; con
      :math:`\Delta t \, P_0 > 1` el modelo deja de ser una probabilidad
      por paso válida y conviene reducir ``TIME_STEP_DT``.
    - ``BETA_RESET`` no interviene en este modelo.

    See Also
    --------
    stochastic_updater1 : modelo previo, sin dependencia en Δt ni en la corriente.
    """
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]

    mem_g = circuit["memristor_g"]
    mem_edges = circuit["mem_edge_keys"]

    dt = p["TIME_STEP_DT"]
    V_th = p["V_THRESHOLD"]
    G_OFF = p["G_OFF"]
    G_ON = p["G_ON"]

    # Snapshot del estado al inicio del paso.
    is_off = mem_g == G_OFF
    is_on = mem_g == G_ON

    v_mem = np.abs(V_solved[circuit["mem_u_idx"]] - V_solved[circuit["mem_v_idx"]])

    # --- SET (OFF -> ON): activado por voltaje ---
    set_candidates = np.flatnonzero(is_off & (v_mem > V_th))
    if set_candidates.size > 0:
        overdrive = v_mem[set_candidates] - V_th
        p_set = dt * p["P0_SET"] * (-np.expm1(-p["ALPHA_SET"] * overdrive))
        p_set = np.clip(p_set, 0.0, 1.0)
        switched = set_candidates[np.random.random(set_candidates.size) < p_set]
        _apply_switch(mem_g, G, mem_edges, switched, G_ON)

    # --- RESET (ON -> OFF): activado por corriente (disipación Joule) ---
    reset_candidates = np.flatnonzero(is_on)
    if reset_candidates.size > 0:
        i_mem = G_ON * v_mem[reset_candidates]
        p_reset = np.minimum(1.0, dt * p["P_DECAY"] * i_mem**2)
        switched = reset_candidates[np.random.random(reset_candidates.size) < p_reset]
        _apply_switch(mem_g, G, mem_edges, switched, G_OFF)

    circuit["memristor_active"] = mem_g == G_ON
    return G


def _apply_switch(mem_g, G, mem_edges, indices, value: float) -> None:
    """Fija ``value`` en ``mem_g[indices]`` y en el espejo del grafo."""
    mem_g[indices] = value
    for j in indices:
        u, v = mem_edges[j]
        G.edges[u, v]["conductance"] = value
