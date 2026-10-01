# geometria.py
from typing import Any

import numpy as np


# ==============================================================================
# FUNCIONES DE GEOMETRÍA Y RED
# ==============================================================================
def generate_and_find_junctions(simulation: dict[str, Any]) -> None:
    """Genera la disposición espacial de los nanohilos y encuentra sus cruces.

    Coloca ``NUM_WIRES`` hilos con centros uniformemente distribuidos en un
    área cuadrada de lado ``AREA`` y orientaciones uniformes en
    :math:`[0, \\pi)`. Detecta los cruces entre pares de segmentos por
    intersección analítica (fuerza bruta :math:`O(N^2)`).

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"`` con las
        claves ``NUM_WIRES``, ``AREA`` y ``LENGTH``.

    Returns
    -------
    None
        Modifica ``simulation`` in-place agregando la clave ``"junctions"``,
        un diccionario con:

        - ``wires`` : list of dict
            Cada entrada contiene ``id`` (int), ``p1`` y ``p2`` (``ndarray``
            de shape ``(2,)``) con las coordenadas de los extremos.
        - ``junctions`` : list of dict
            Cada entrada contiene ``id`` (int), ``pos`` (``ndarray`` de
            shape ``(2,)``) y ``wires`` (``tuple[int, int]``) con los IDs
            de los dos hilos que se cruzan.
        - ``wire_to_junctions`` : dict of list
            Mapa de adyacencia que asocia cada ID de hilo con la lista de
            junturas que lo atraviesan.

    Notes
    -----
    La detección por fuerza bruta es aceptable para :math:`N \\lesssim 10^4`.
    Para redes más densas conviene vectorizar el cálculo de intersecciones
    con NumPy.
    """
    # Extraemos las variables del diccionario centralizado
    p: dict[str, Any] = simulation["parameters"]
    num_wires: int = p["NUM_WIRES"]
    wire_length: float = p["LENGTH"]
    area_size: float = p["AREA"]

    # Generación de posiciones aleatorias basándonos en el substrato
    xc = np.random.uniform(0, area_size, num_wires)
    yc = np.random.uniform(0, area_size, num_wires)
    theta = np.random.uniform(0, np.pi, num_wires)

    x_off = (wire_length / 2) * np.cos(theta)
    y_off = (wire_length / 2) * np.sin(theta)

    wires: list[dict[str, Any]] = []
    for i in range(num_wires):
        wires.append(
            {
                "id": i,
                "p1": np.array((xc[i] - x_off[i], yc[i] - y_off[i])),
                "p2": np.array((xc[i] + x_off[i], yc[i] + y_off[i])),
            }
        )

    junctions: list[dict[str, Any]] = []
    wire_to_junctions: dict[int, list[dict[str, Any]]] = {i: [] for i in range(num_wires)}
    junction_id_counter = 0

    # Detección de intersecciones por fuerza bruta
    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            w1, w2 = wires[i], wires[j]
            d1 = w1["p2"] - w1["p1"]
            d2 = w2["p2"] - w2["p1"]
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if denom != 0:
                t = (
                    (w2["p1"][0] - w1["p1"][0]) * d2[1] - (w2["p1"][1] - w1["p1"][1]) * d2[0]
                ) / denom
                u = (
                    (w2["p1"][0] - w1["p1"][0]) * d1[1] - (w2["p1"][1] - w1["p1"][1]) * d1[0]
                ) / denom
                if 0 <= t <= 1 and 0 <= u <= 1:
                    ix, iy = w1["p1"] + t * d1
                    j_data = {"id": junction_id_counter, "pos": np.array([ix, iy]), "wires": (i, j)}
                    junctions.append(j_data)
                    wire_to_junctions[i].append(j_data)
                    wire_to_junctions[j].append(j_data)
                    junction_id_counter += 1

    simulation["junctions"] = {
        "wires": wires,
        "junctions": junctions,
        "wire_to_junctions": wire_to_junctions,
    }
    return
