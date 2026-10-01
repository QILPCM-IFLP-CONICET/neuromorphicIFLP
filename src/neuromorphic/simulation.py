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
    """Construye el diccionario completo de simulación.

    Encadena la carga de parámetros, la generación geométrica de la red,
    la construcción del grafo, la detección de electrodos y el ensamblado
    inicial de la matriz de admitancia.

    Parameters
    ----------
    filepath : str, optional
        Ruta a un ``.ini`` explícito. Si es ``None``, se usa el
        ``defaults.ini`` empaquetado.
    parms : dict, optional
        Overrides sobre los parámetros cargados. Las claves desconocidas
        disparan :class:`~neuromorphic.load_config.UnknownParameterWarning`
        y se ignoran.

    Returns
    -------
    dict
        Diccionario con las claves:

        - ``"parameters"`` : parámetros crudos y derivados.
        - ``"junctions"`` : estructuras geométricas de la red.
        - ``"graph"`` : objeto :class:`networkx.Graph`.
        - ``"terminals"`` : ``{"input_nodes", "output_nodes"}``.
        - ``"circuit"`` : ``{"Y", "I", "node_to_index"}``.

    See Also
    --------
    cargar_parametros : carga y valida los parámetros.
    run_simulation_dynamic_pulse : corre el experimento de pulso.
    """
    simulation = cargar_parametros(filepath, parms=parms)
    generate_and_find_junctions(simulation)
    build_graph2(simulation)
    find_electrode_nodes2(simulation)
    build_admittance_matrix2(simulation)
    return simulation
