# simulation.py
"""High-level entry point that assembles a simulation dict."""
from __future__ import annotations

from typing import Any

from .fisica import build_admittance_matrix2
from .geometria import generate_and_find_junctions
from .grafo import build_graph2, find_electrode_nodes2
from .load_config import cargar_parametros


def setup_simulation(
    filepath: str | None = None,
    parms: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a simulation dict from config + overrides.

    Parameters
    ----------
    filepath : str, optional
        Path to an explicit ``.ini`` file. If ``None``, the packaged
        defaults are used.
    parms : dict, optional
        Parameter overrides applied on top of the file. Unknown keys
        emit an ``UnknownParameterWarning`` and are ignored.

    Returns
    -------
    dict
        Keys: ``parameters``, ``junctions``, ``graph``, ``terminals``,
        ``circuit``.
    """
    simulation = cargar_parametros(filepath, parms=parms)
    generate_and_find_junctions(simulation)
    build_graph2(simulation)
    find_electrode_nodes2(simulation)
    build_admittance_matrix2(simulation)
    return simulation
