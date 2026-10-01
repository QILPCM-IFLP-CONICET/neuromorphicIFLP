# grafo.py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ==============================================================================
# CONSTRUCCIÓN DEL GRAFO
# ==============================================================================
def build_graph2(simulation: dict):
    '''
    Construye un grafo a partir de una lista de nanohilos y un diccionario de uniones,
    duplicando nodos para representar memristores como aristas entre los nodos duplicados.
    Cada unión original J se convierte en dos nodos en el grafo (J_node_W1, J_node_W2).
    El nodo J_node_W1 (impar) está asociado con el primer nanohilo de la unión (wires[0]).
    El nodo J_node_W2 (par) está asociado con el segundo nanohilo de la unión (wires[1]).
    Un memristor se coloca como una arista entre J_node_W1 y J_node_W2.
    Los segmentos de nanohilos (resistencias) conectan nodos que representan
    el mismo nanohilo en uniones adyacentes.
    '''
    p = simulation["parameters"]
    junction_data = simulation["junctions"]
    wires = junction_data["wires"]
    junctions = junction_data["junctions"]
    wire_to_junctions = junction_data["wire_to_junctions"]

    G = nx.Graph()
    wire_junction_to_graph_node = {}

    # Crear dos nodos por unión y la arista memristiva entre ellos
    for j_data in junctions:
        orig_junction_id = j_data['id']
        w1_id, w2_id = j_data['wires']

        node_for_w1_side = 2 * orig_junction_id + 1
        node_for_w2_side = 2 * orig_junction_id

        wire_junction_to_graph_node[(orig_junction_id, w1_id)] = node_for_w1_side
        wire_junction_to_graph_node[(orig_junction_id, w2_id)] = node_for_w2_side

        G.add_node(node_for_w1_side, pos=j_data['pos'])
        G.add_node(node_for_w2_side, pos=j_data['pos'])

        G.add_edge(node_for_w1_side, node_for_w2_side,
                   is_memristor=True, conductance=p['G_OFF'])

    # Segmentos internos de cada nanohilo (resistencias lineales)
    for wire_id, junctions_list_for_wire in wire_to_junctions.items():
        if len(junctions_list_for_wire) < 2:
            continue

        start_point = wires[wire_id]['p1']
        junctions_with_dist = []
        for j in junctions_list_for_wire:
            dist = np.linalg.norm(j['pos'] - start_point)
            junctions_with_dist.append((dist, j))
        junctions_with_dist.sort(key=lambda x: x[0])

        for k in range(len(junctions_with_dist) - 1):
            u_orig_junction_data = junctions_with_dist[k][1]
            v_orig_junction_data = junctions_with_dist[k + 1][1]
            dist_between = junctions_with_dist[k + 1][0] - junctions_with_dist[k][0]

            u_graph_node = wire_junction_to_graph_node[(u_orig_junction_data['id'], wire_id)]
            v_graph_node = wire_junction_to_graph_node[(v_orig_junction_data['id'], wire_id)]

            G.add_edge(u_graph_node, v_graph_node, weight=dist_between)

    simulation["graph"] = G
    return


# ==============================================================================
# DETECCIÓN DE ELECTRODOS
# ==============================================================================
def find_electrode_nodes2(simulation: dict):
    ''' Encuentra los nodos conectados a los electrodos. '''
    G = simulation["graph"]
    p = simulation["parameters"]
    area_size = p['AREA']
    threshold = p['PROXIMITY_THRESHOLD']

    pos = nx.get_node_attributes(G, 'pos')
    input_nodes, output_nodes = [], []
    for node_id, coords in pos.items():
        x = coords[0]
        if x < threshold:
            input_nodes.append(node_id)
        elif x > area_size - threshold:
            output_nodes.append(node_id)

    simulation["terminals"] = {
        "input_nodes": input_nodes,
        "output_nodes": output_nodes,
    }
    return


# ==============================================================================
# FUNCIONES DE PERCOLACIÓN Y CAMINOS
# ==============================================================================
def check_percolation(simulation: dict):
    """ Verifica si existe al menos un camino que conecte entrada con salida. """
    G = simulation["graph"]
    input_nodes = simulation["terminals"]["input_nodes"]
    output_nodes = simulation["terminals"]["output_nodes"]

    for in_node in input_nodes:
        for out_node in output_nodes:
            if nx.has_path(G, in_node, out_node):
                return True
    return False


def get_shortest_path_length(simulation: dict):
    """ Calcula la longitud topológica del camino más corto entre electrodos. """
    G = simulation["graph"]
    input_nodes = simulation["terminals"]["input_nodes"]
    output_nodes = simulation["terminals"]["output_nodes"]

    best_length = float('inf')
    for in_node in input_nodes:
        for out_node in output_nodes:
            try:
                length = nx.shortest_path_length(G, source=in_node, target=out_node)
                if length < best_length:
                    best_length = length
            except nx.NetworkXNoPath:
                continue
    return best_length if best_length != float('inf') else None