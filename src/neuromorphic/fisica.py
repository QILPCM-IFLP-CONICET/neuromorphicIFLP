# fisica.py
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix


# ==============================================================================
# CONSTRUCCIÓN DE LA MATRIZ DE ADMITANCIA (SPARSE)
# ==============================================================================
def build_admittance_matrix(simulation: dict[str, Any], v_input: float | None = None):
    """Ensambla la matriz de admitancia y el vector de corrientes.

    Construye el sistema :math:`Y \\cdot V = I` aplicando las leyes de
    Kirchhoff sobre todos los nodos internos y condiciones de contorno
    de Dirichlet en los electrodos. Agrega una conductancia de fuga
    mínima (``G_LEAK``) en los nodos internos para evitar singularidades
    numéricas.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"`` y ``"terminals"``.
    v_input : float, optional
        Voltaje de entrada a fijar en esta llamada. Si es ``None``
        (default), se usa ``parameters["V_INPUT"]``. Se usa en
        simulaciones dinámicas para no mutar el estado persistente.

    Returns
    -------
    Y : scipy.sparse.csr_matrix
        Matriz de admitancia dispersa de shape ``(N, N)``.
    I_vec : numpy.ndarray
        Vector de corrientes externas de shape ``(N,)``.
    node_to_index : dict
        Mapeo ``node_id -> row_index``.

    Notes
    -----
    También deja el resultado disponible en
    ``simulation["circuit"]`` como ``{"Y", "I", "node_to_index"}``.

    El código asume que la red percola (ver :func:`check_percolation`). Con
    una red no percolante y ``G_LEAK`` muy pequeño, la matriz puede quedar
    numéricamente singular.

    """
    p = simulation["parameters"]
    G = simulation["graph"]
    terminals = simulation["terminals"]
    input_nodes = terminals["input_nodes"]
    output_nodes = terminals["output_nodes"]

    if v_input is None:
        v_input = p["V_INPUT"]
    v_ground = p["V_GROUND"]

    N = G.number_of_nodes()
    node_to_index = {node: i for i, node in enumerate(G.nodes)}

    # Usamos lil_matrix para el llenado eficiente
    Y = lil_matrix((N, N))
    I_vec = np.zeros(N)

    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_to_index[u], node_to_index[v]

        if data.get("is_memristor", False):
            # Lee G_OFF desde el diccionario
            conductance = data.get("conductance", p["G_OFF"])
        else:
            # Lee R_WIRE_PER_LENGTH calculado con pi desde el diccionario
            conductance = 1.0 / (data["weight"] * p["R_WIRE_PER_LENGTH"] + 1e-12)

        Y[u_idx, v_idx] -= conductance
        Y[v_idx, u_idx] -= conductance
        Y[u_idx, u_idx] += conductance
        Y[v_idx, v_idx] += conductance

    # --- Condiciones de contorno de Dirichlet (fijan voltaje en electrodos de entrada y salida) ---
    electrode_idx = set()

    for node_id in input_nodes:
        idx = node_to_index[node_id]
        Y[idx, :] = 0.0
        Y[idx, idx] = 1.0
        I_vec[idx] = v_input
        electrode_idx.add(idx)

    for node_id in output_nodes:
        idx = node_to_index[node_id]
        Y[idx, :] = 0.0
        Y[idx, idx] = 1.0
        I_vec[idx] = v_ground
        electrode_idx.add(idx)

    # --- Estabilización numérica: fuga mínima a tierra en nodos internos ---
    # (NO se aplica a las filas de los electrodos, para no alterar los BCs)
    g_leak = p["G_LEAK"]
    for i in range(N):
        if i not in electrode_idx:
            Y[i, i] += g_leak

    # Convertimos a CSR (Compressed Sparse Row) para que el solver vuele
    Y_csr = Y.tocsr()
    circuit_data = {
        "Y": Y_csr,
        "I": I_vec,
        "node_to_index": node_to_index,
    }
    simulation["circuit"] = circuit_data
    return Y_csr, I_vec, node_to_index


# ==============================================================================
# ACTUALIZACIÓN ESTOCÁSTICA DE MEMRISTORES
# ==============================================================================
def update_stochastic_conductance(simulation: dict[str, Any], V_solved):
    """Actualiza estocásticamente el estado de los memristores.

    Evalúa el voltaje local :math:`V_{\\text{mem}} = |V_u - V_v|` en cada
    arista memristiva y aplica las reglas de conmutación:

    - **SET** (OFF -> ON): si :math:`V_{\\text{mem}} > V_{\\text{threshold}}`,
      con probabilidad exponencial en :math:`|V_{\\text{mem}} - V_{\\text{threshold}}|`.
    - **RESET** (ON -> OFF): con probabilidad modulada por un factor de
      estabilidad :math:`\\exp(-|V_{\\text{mem}}| / V_{\\text{threshold}})`.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto, indexado por
        ``simulation["circuit"]["node_to_index"]``.

    Returns
    -------
    networkx.Graph
        El mismo objeto ``simulation["graph"]`` mutado in-place.

    See Also
    --------
    build_admittance_matrix2 : construye el sistema que produce ``V_solved``.
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    node_to_index = simulation["circuit"]["node_to_index"]

    edges_to_set = []
    edges_to_reset = []

    for u, v, data in G.edges(data=True):
        if not data.get("is_memristor", False):
            continue

        u_idx, v_idx = node_to_index[u], node_to_index[v]
        V_mem = abs(V_solved[u_idx] - V_solved[v_idx])
        current_G = data.get("conductance", p["G_OFF"])

        # --- SET (Facilitación) ---
        if current_G == p["G_OFF"]:
            if V_mem > p["V_THRESHOLD"]:
                # Probabilidad exponencial basada en la activación iónica (Ag+)
                p_set = p["P0_SET"] * np.exp(p["ALPHA_SET"] * np.abs(V_mem - p["V_THRESHOLD"]))
                if np.random.rand() < p_set:
                    edges_to_set.append((u, v))

        # --- RESET (Relajación volátil) ---
        elif current_G == p["G_ON"]:
            #  Factor de estabilidad: disminuye el decay si hay voltaje suficiente
            estabilidad = np.exp(-np.abs(V_mem) / p["V_THRESHOLD"])
            p_decay = p["P_DECAY"] * estabilidad
            if np.random.rand() < p_decay:
                edges_to_reset.append((u, v))

    # Aplicar cambios en lote
    for u, v in edges_to_set:
        G.edges[u, v]["conductance"] = p["G_ON"]
    for u, v in edges_to_reset:
        G.edges[u, v]["conductance"] = p["G_OFF"]

    return G


# ==============================================================================
# CÁLCULOS DE CORRIENTES
# ==============================================================================
def _edge_conductance(data, p):
    """Conductancia de una arista: memristor (dinámica) o segmento resistivo."""
    if data.get("is_memristor", False):
        return data.get("conductance", p["G_OFF"])
    return 1.0 / (data["weight"] * p["R_WIRE_PER_LENGTH"] + 1e-12)


def calculate_input_current(simulation: dict[str, Any], V_solved) -> float:
    """Corriente neta inyectada por los electrodos de entrada.

    Suma las contribuciones :math:`(V_{\\text{in}} - V_{\\text{vecino}}) \\cdot G`
    sobre cada arista que conecta un nodo de entrada con un nodo fuera
    del conjunto de entradas.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"``, ``"terminals"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto.

    Returns
    -------
    float
        Corriente neta en amperios (positiva si entra a la red).
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    node_to_index = simulation["circuit"]["node_to_index"]
    input_nodes = simulation["terminals"]["input_nodes"]
    input_set = set(input_nodes)

    total = 0.0
    for in_node in input_nodes:
        V_in = V_solved[node_to_index[in_node]]
        for neighbor in G.neighbors(in_node):
            if neighbor in input_set:
                continue
            data = G.get_edge_data(in_node, neighbor)
            g = _edge_conductance(data, p)
            V_n = V_solved[node_to_index[neighbor]]
            total += (V_in - V_n) * g
    return total


def calculate_output_current(simulation: dict[str, Any], V_solved) -> float:
    """Corriente neta recolectada por los electrodos de salida.

    Suma las contribuciones :math:`(V_{\\text{vecino}} - V_{\\text{out}}) \\cdot G`
    sobre cada arista que conecta un nodo de salida con un nodo fuera
    del conjunto de salidas.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"``, ``"terminals"`` y ``"circuit"``.
    V_solved : numpy.ndarray
        Vector de voltajes nodales resuelto.

    Returns
    -------
    float
        Corriente neta en amperios.
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    node_to_index = simulation["circuit"]["node_to_index"]
    output_nodes = simulation["terminals"]["output_nodes"]
    output_set = set(output_nodes)

    total = 0.0
    for out_node in output_nodes:
        V_out = V_solved[node_to_index[out_node]]
        for neighbor in G.neighbors(out_node):
            if neighbor in output_set:
                continue
            data = G.get_edge_data(out_node, neighbor)
            g = _edge_conductance(data, p)
            V_n = V_solved[node_to_index[neighbor]]
            total += (V_n - V_out) * g
    return total


# ==============================================================================
# SEÑALES DE ENTRADA EXTRA
# ==============================================================================
def get_v_ramp(t: float, total_time: float, amplitude: float) -> float:
    """Señal senoidal para barridos de histéresis.

    Parameters
    ----------
    t : float
        Tiempo actual.
    total_time : float
        Período completo de la señal.
    amplitude : float
        Amplitud pico (V).

    Returns
    -------
    float
        Voltaje instantáneo :math:`V_0 \\sin(2 \\pi t / T)`.
    """
    return amplitude * np.sin(2 * np.pi * t / total_time)
