# admitancia.py
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu


# ==============================================================================
# ELECTRODOS
# ==============================================================================
def _electrode_node_lists(simulation: dict[str, Any]) -> tuple[list[str], list[list[Any]]]:
    """Nombres y nodos de cada electrodo, en el orden que define su índice.

    Mientras ``terminals`` solo tenga el par entrada/salida, se interpretan
    como dos electrodos: ``0 = "input"`` y ``1 = "output"``. Cuando exista
    ``terminals["electrodes"]`` (lista de dicts con ``"name"`` y
    ``"nodes"``), se usa esa lista y su orden.
    """
    terminals = simulation["terminals"]
    if "electrodes" in terminals:
        electrodes = terminals["electrodes"]
        return [e["name"] for e in electrodes], [list(e["nodes"]) for e in electrodes]
    return (
        ["input", "output"],
        [list(terminals["input_nodes"]), list(terminals["output_nodes"])],
    )


def _build_electrode_of_node(
    node_lists: list[list[Any]],
    names: list[str],
    node_to_index: dict[Any, int],
    N: int,
) -> np.ndarray:
    """Array ``(N,)`` con el índice de electrodo de cada nodo (``-1`` si es interno).

    Raises
    ------
    ValueError
        Si un nodo pertenece a más de un electrodo.
    """
    electrode_of_node = np.full(N, -1, dtype=np.int64)
    for k, nodes in enumerate(node_lists):
        idx = np.fromiter((node_to_index[n] for n in nodes), dtype=np.int64, count=len(nodes))
        taken = electrode_of_node[idx]
        clash = taken >= 0
        if clash.any():
            other = int(taken[clash][0])
            raise ValueError(
                f"{int(clash.sum())} nodo(s) pertenecen a la vez a los electrodos "
                f"'{names[other]}' y '{names[k]}'. Las regiones de los electrodos "
                "no pueden superponerse."
            )
        electrode_of_node[idx] = k
    return electrode_of_node


# ==============================================================================
# PRECÓMPUTO DE ESTRUCTURAS (una vez por simulación)
# ==============================================================================
def _precompute_circuit_arrays(simulation: dict[str, Any]) -> None:
    """Cachea arrays con la estructura del grafo para evitar loops Python.

    Calcula y guarda en ``simulation["circuit"]``:

    - ``node_to_index``: mapeo nodo -> índice.
    - ``edge_u_idx``, ``edge_v_idx``: índices de los extremos por arista.
    - ``edge_is_memristor``: máscara booleana.
    - ``edge_fixed_g``: conductancia fija por arista (0 en memristores).
    - ``mem_edge_idx``: índices (en el array de aristas) de los memristores.
    - ``memristor_g``: conductancias actuales de los memristores.
    - ``input_idx``, ``output_idx``: índices de los electrodos.
    - ``electrode_of_node``: array ``(N,)`` con el índice de electrodo de
      cada nodo, ``-1`` para los nodos internos.
    - ``n_electrodes``, ``electrode_names``: cantidad y nombres de los
      electrodos; el índice ``k`` de ``electrode_of_node`` corresponde a
      ``electrode_names[k]``.
    - ``unknown_of_node``: array ``(N,)`` que asigna a cada nodo su fila en
      el sistema reducido (ver :func:`build_reduced_admittance_matrix`).
      Los nodos internos ocupan las filas ``0 .. n_internal-1`` en el orden
      de ``node_to_index``; todos los nodos del electrodo ``k`` comparten la
      fila ``n_internal + k``.
    - ``n_internal``: cantidad de nodos internos.
    - ``reduced_edge_mask``: aristas que conectan filas distintas del sistema
      reducido (excluye las que unen dos nodos del mismo electrodo).
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    circuit = simulation["circuit"]

    nodes = list(G.nodes)
    node_to_index = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)

    edges = list(G.edges(data=True))
    E = len(edges)

    u_idx = np.empty(E, dtype=np.int64)
    v_idx = np.empty(E, dtype=np.int64)
    is_mem = np.empty(E, dtype=bool)
    fixed_g = np.empty(E, dtype=np.float64)

    mem_count = 0
    for i, (u, v, data) in enumerate(edges):
        u_idx[i] = node_to_index[u]
        v_idx[i] = node_to_index[v]
        mem = data.get("is_memristor", False)
        is_mem[i] = mem
        if mem:
            fixed_g[i] = 0.0
            data["_mem_idx"] = mem_count  # índice del memristor en memristor_g
            mem_count += 1
        else:
            fixed_g[i] = 1.0 / (data["weight"] * p["R_WIRE_PER_LENGTH"] + 1e-12)

    mem_edge_idx = np.flatnonzero(is_mem)
    mem_g = np.empty(mem_count, dtype=np.float64)
    for j, edge_pos in enumerate(mem_edge_idx):
        mem_g[j] = edges[edge_pos][2]["conductance"]

    input_idx = np.array(
        [node_to_index[n] for n in simulation["terminals"]["input_nodes"]],
        dtype=np.int64,
    )
    output_idx = np.array(
        [node_to_index[n] for n in simulation["terminals"]["output_nodes"]],
        dtype=np.int64,
    )

    electrode_names, electrode_nodes = _electrode_node_lists(simulation)
    electrode_of_node = _build_electrode_of_node(electrode_nodes, electrode_names, node_to_index, N)

    is_internal = electrode_of_node < 0
    n_internal = int(is_internal.sum())
    unknown_of_node = np.where(is_internal, -1, n_internal + electrode_of_node)
    unknown_of_node[is_internal] = np.arange(n_internal, dtype=np.int64)
    reduced_edge_mask = unknown_of_node[u_idx] != unknown_of_node[v_idx]

    circuit["node_to_index"] = node_to_index
    circuit["N"] = N
    circuit["edge_u_idx"] = u_idx
    circuit["edge_v_idx"] = v_idx
    circuit["edge_is_memristor"] = is_mem
    circuit["edge_fixed_g"] = fixed_g
    circuit["mem_edge_idx"] = mem_edge_idx
    circuit["memristor_g"] = mem_g
    circuit["mem_u_idx"] = u_idx[mem_edge_idx]
    circuit["mem_v_idx"] = v_idx[mem_edge_idx]
    circuit["mem_edge_keys"] = [edges[i][:2] for i in mem_edge_idx]
    circuit["input_idx"] = input_idx
    circuit["output_idx"] = output_idx
    circuit["electrode_of_node"] = electrode_of_node
    circuit["n_electrodes"] = len(electrode_names)
    circuit["electrode_names"] = electrode_names
    circuit["unknown_of_node"] = unknown_of_node
    circuit["n_internal"] = n_internal
    circuit["reduced_edge_mask"] = reduced_edge_mask


def _edge_conductances(circuit: dict[str, Any]) -> np.ndarray:
    """Conductancia actual de cada arista: segmentos fijos + memristores."""
    edge_g = circuit["edge_fixed_g"].copy()
    mem_edge_idx = circuit["mem_edge_idx"]
    if len(mem_edge_idx) > 0:
        edge_g[mem_edge_idx] = circuit["memristor_g"]
    return edge_g


def _laplacian_triplets(
    u_idx: np.ndarray, v_idx: np.ndarray, edge_g: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Triplets COO del laplaciano pesado: 4 entradas por arista."""
    E = len(u_idx)
    rows = np.empty(4 * E, dtype=np.int64)
    cols = np.empty(4 * E, dtype=np.int64)
    vals = np.empty(4 * E, dtype=np.float64)

    rows[0::4] = u_idx
    cols[0::4] = u_idx
    vals[0::4] = edge_g

    rows[1::4] = v_idx
    cols[1::4] = v_idx
    vals[1::4] = edge_g

    rows[2::4] = u_idx
    cols[2::4] = v_idx
    vals[2::4] = -edge_g

    rows[3::4] = v_idx
    cols[3::4] = u_idx
    vals[3::4] = -edge_g
    return rows, cols, vals


# ==============================================================================
# CONSTRUCCIÓN DE LA MATRIZ DE ADMITANCIA (SPARSE)
# ==============================================================================
def build_admittance_matrix(
    simulation: dict[str, Any],
    v_input: float | None = None,
):
    """Ensambla la matriz de admitancia y el vector de corrientes.

    Construye el sistema :math:`Y \\cdot V = I` en formato COO → CSR
    usando arrays de NumPy vectorizados. La estructura del grafo se
    precalcula una sola vez y se reutiliza en cada paso.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación. Debe contener ``"parameters"``,
        ``"graph"`` y ``"terminals"``.
    v_input : float, optional
        Voltaje de entrada a fijar en esta llamada. Si es ``None``
        (default), se usa ``parameters["V_INPUT"]``.

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
    También deja el resultado disponible en ``simulation["circuit"]``
    como ``{"Y", "I", "node_to_index", ...}``.

    El código asume que la red percola y que ``prune_dead_components``
    fue ejecutado. Con nodos flotantes y ``G_LEAK`` pequeño, la matriz
    puede quedar numéricamente singular.
    """
    p = simulation["parameters"]
    if "circuit" not in simulation:
        simulation["circuit"] = {}
    circuit = simulation["circuit"]

    if "edge_u_idx" not in circuit:
        _precompute_circuit_arrays(simulation)

    if v_input is None:
        v_input = p["V_INPUT"]
    v_ground = p["V_GROUND"]
    g_leak = p["G_LEAK"]

    u_idx = circuit["edge_u_idx"]
    v_idx = circuit["edge_v_idx"]
    N = circuit["N"]
    input_idx = circuit["input_idx"]
    output_idx = circuit["output_idx"]

    # --- Laplaciano pesado (segmentos fijos + memristores actuales) ---
    rows, cols, vals = _laplacian_triplets(u_idx, v_idx, _edge_conductances(circuit))

    # --- G_LEAK en diagonales de nodos internos ---
    electrode_mask = np.zeros(N, dtype=bool)
    electrode_mask[input_idx] = True
    electrode_mask[output_idx] = True

    leak_rows = np.arange(N, dtype=np.int64)
    leak_vals = np.where(electrode_mask, 0.0, g_leak)

    all_rows = np.concatenate([rows, leak_rows])
    all_cols = np.concatenate([cols, leak_rows])
    all_vals = np.concatenate([vals, leak_vals])

    Y = coo_matrix((all_vals, (all_rows, all_cols)), shape=(N, N)).tocsr()

    # --- Condiciones de contorno de Dirichlet (fila identidad) ---
    I_vec = np.zeros(N)

    for idx in input_idx:
        s, e = Y.indptr[idx], Y.indptr[idx + 1]
        cols_in_row = Y.indices[s:e]
        Y.data[s:e] = np.where(cols_in_row == idx, 1.0, 0.0)
        I_vec[idx] = v_input

    for idx in output_idx:
        s, e = Y.indptr[idx], Y.indptr[idx + 1]
        cols_in_row = Y.indices[s:e]
        Y.data[s:e] = np.where(cols_in_row == idx, 1.0, 0.0)
        I_vec[idx] = v_ground

    Y.eliminate_zeros()

    circuit["Y"] = Y
    circuit["I"] = I_vec
    return Y, I_vec, circuit["node_to_index"]


# ==============================================================================
# MATRIZ DE ADMITANCIA REDUCIDA (UNA INCÓGNITA POR ELECTRODO)
# ==============================================================================
def build_reduced_admittance_matrix(simulation: dict[str, Any]) -> csr_matrix:
    """Ensambla la matriz de admitancia con cada electrodo contraído a un nodo.

    Todos los nodos de contacto del electrodo ``k`` se identifican con una
    única fila, ``circuit["n_internal"] + k``; los nodos internos conservan
    una fila propia (ver ``circuit["unknown_of_node"]``). El resultado es el
    laplaciano pesado de la red con los electrodos contraídos, más
    ``G_LEAK`` en la diagonal de los nodos internos:

    .. math::

        Y_{\\mathrm{red}} = P^{T} \\, (L + G_{\\mathrm{leak}} D_{\\mathrm{int}}) \\, P,

    con :math:`P_{n m} = 1` si el nodo :math:`n` corresponde a la fila
    :math:`m`. Fijar el mismo voltaje en todos los contactos de un electrodo,
    como hace :func:`build_admittance_matrix`, es equivalente a esta
    contracción.

    No se imponen condiciones de contorno: la matriz es simétrica y las
    filas de los electrodos suman cero. Las aristas que unen dos nodos del
    mismo electrodo no conducen corriente y se omiten.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación con ``"parameters"``, ``"graph"`` y
        ``"terminals"``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Matriz de shape ``(M, M)`` con ``M = n_internal + n_electrodes``.
        Las primeras ``n_internal`` filas son los nodos internos y las
        últimas ``n_electrodes``, los electrodos en el orden de
        ``circuit["electrode_names"]``.
    """
    if "circuit" not in simulation:
        simulation["circuit"] = {}
    circuit = simulation["circuit"]
    if "unknown_of_node" not in circuit:
        _precompute_circuit_arrays(simulation)

    n_internal = circuit["n_internal"]
    M = n_internal + circuit["n_electrodes"]
    m_of = circuit["unknown_of_node"]
    keep = circuit["reduced_edge_mask"]

    edge_g = _edge_conductances(circuit)[keep]
    rows, cols, vals = _laplacian_triplets(
        m_of[circuit["edge_u_idx"][keep]], m_of[circuit["edge_v_idx"][keep]], edge_g
    )

    leak_rows = np.arange(n_internal, dtype=np.int64)
    leak_vals = np.full(n_internal, simulation["parameters"]["G_LEAK"])

    Y_red = coo_matrix(
        (
            np.concatenate([vals, leak_vals]),
            (np.concatenate([rows, leak_rows]), np.concatenate([cols, leak_rows])),
        ),
        shape=(M, M),
    ).tocsr()
    return Y_red


# ==============================================================================
# RESOLUCIÓN DEL SISTEMA PARTICIONADO
# ==============================================================================
def _solve_partitioned(Y_red: csr_matrix, known: np.ndarray, V_known: np.ndarray) -> np.ndarray:
    """Resuelve ``Y_red V = 0`` en las filas libres con ``V[known] = V_known``.

    Separa las filas en conocidas (``b``) y libres (``u``) y resuelve

    .. math::

        Y_{uu} \\, V_u = -Y_{ub} \\, V_b .

    ``Y_uu`` es simétrica y definida positiva cuando cada componente conexa
    toca al menos una fila conocida, así que se factoriza con un
    ordenamiento simétrico.

    Returns
    -------
    numpy.ndarray
        Vector ``V`` completo de shape ``(M,)``.
    """
    M = Y_red.shape[0]
    is_known = np.zeros(M, dtype=bool)
    is_known[known] = True
    free = np.flatnonzero(~is_known)

    Y = Y_red.tocsc()
    Y_uu = Y[free][:, free]
    Y_ub = Y[free][:, known]

    V = np.empty(M)
    V[known] = V_known
    if free.size:
        lu = splu(Y_uu, permc_spec="MMD_AT_PLUS_A", options={"SymmetricMode": True})
        V[free] = lu.solve(-(Y_ub @ V_known))
    return V


def solve_node_voltages(simulation: dict[str, Any], V_b: Any) -> np.ndarray:
    """Voltajes nodales con un voltaje fijado en cada electrodo.

    Ensambla la matriz reducida (:func:`build_reduced_admittance_matrix`),
    fija la fila de cada electrodo en el valor dado y resuelve Kirchhoff
    para los nodos internos.

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación.
    V_b : array_like
        Voltajes de los electrodos, uno por electrodo, en el orden de
        ``circuit["electrode_names"]``.

    Returns
    -------
    numpy.ndarray
        Vector de voltajes de shape ``(N,)``, indexado como
        ``circuit["node_to_index"]``. Todos los contactos de un electrodo
        tienen exactamente el voltaje de ese electrodo.

    Raises
    ------
    ValueError
        Si ``V_b`` no tiene un valor por electrodo.

    Notes
    -----
    Cuando partes extensas de la red quedan unidas a los electrodos solo por
    memristores en ``G_OFF``, el contraste con la conductancia de los
    segmentos de hilo (de ``1e7`` o más) limita la precisión de los voltajes
    nodales en esas regiones a ~``1e-5`` relativo en doble precisión. El
    residuo de Kirchhoff, en cambio, queda en el orden del error de redondeo.
    """
    Y_red = build_reduced_admittance_matrix(simulation)
    circuit = simulation["circuit"]
    n_internal = circuit["n_internal"]
    n_electrodes = circuit["n_electrodes"]

    V_b = np.asarray(V_b, dtype=np.float64)
    if V_b.shape != (n_electrodes,):
        raise ValueError(
            f"V_b debe tener un valor por electrodo: se esperaban {n_electrodes} "
            f"({circuit['electrode_names']}), se recibió shape {V_b.shape}."
        )

    known = np.arange(n_internal, n_internal + n_electrodes)
    V_red = _solve_partitioned(Y_red, known, V_b)
    return V_red[circuit["unknown_of_node"]]
