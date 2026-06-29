import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

AREA = 1000.0
PROXIMITY_THRESHOLD = 0.05 * AREA
G_OFF = 1.0 / 1e9

def build_graph2(wires, junctions, wire_to_junctions, G_OFF_default=G_OFF):
    """ Construye el grafo mapeando memristores como aristas entre nodos duplicados """
    G = nx.Graph()
    wire_junction_to_graph_node = {}

    for j_data in junctions:
        orig_junction_id = j_data['id']
        w1_id, w2_id = j_data['wires']

        node_for_w1_side = 2 * orig_junction_id + 1
        node_for_w2_side = 2 * orig_junction_id

        wire_junction_to_graph_node[(orig_junction_id, w1_id)] = node_for_w1_side
        wire_junction_to_graph_node[(orig_junction_id, w2_id)] = node_for_w2_side

        G.add_node(node_for_w1_side, pos=j_data['pos'])
        G.add_node(node_for_w2_side, pos=j_data['pos'])
        G.add_edge(node_for_w1_side, node_for_w2_side, is_memristor=True, conductance=G_OFF_default)

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
            v_orig_junction_data = junctions_with_dist[k+1][1]
            dist_between = junctions_with_dist[k+1][0] - junctions_with_dist[k][0]

            u_graph_node = wire_junction_to_graph_node[(u_orig_junction_data['id'], wire_id)]
            v_graph_node = wire_junction_to_graph_node[(v_orig_junction_data['id'], wire_id)]
            G.add_edge(u_graph_node, v_graph_node, weight=dist_between)

    return G

def find_electrode_nodes2(G, area_size=AREA, threshold=PROXIMITY_THRESHOLD):
    """ Identifica los nodos adyacentes a las fronteras izquierda y derecha """
    pos = nx.get_node_attributes(G, 'pos')
    input_nodes, output_nodes = [], []
    for node_id, coords in pos.items():
        x = coords[0]
        if x < threshold: 
            input_nodes.append(node_id)
        elif x > area_size - threshold: 
            output_nodes.append(node_id)
    return input_nodes, output_nodes

def check_percolation(G, input_nodes, output_nodes):
    """ Verifica si existe al menos un camino que conecte la entrada con la salida """
    for in_node in input_nodes:
        for out_node in output_nodes:
            if nx.has_path(G, in_node, out_node):
                return True
    return False

def get_shortest_path_length(G, input_nodes, output_nodes):
    """ Calcula la longitud topológica del camino más corto que conecta los extremos """
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