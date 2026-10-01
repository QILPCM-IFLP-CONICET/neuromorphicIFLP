# grafo.py

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ==============================================================================
# CONSTRUCCIÓN DEL GRAFO
# ==============================================================================
def build_graph2(simulation:dict):
    ''' Construye un grafo a partir de una lista de nanohilos y un diccionario de uniones,
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
    junctions=junction_data["junctions"]
    wire_to_junctions = junction_data["wire_to_junctions"]
    
    G = nx.Graph()

    # wire_junction_to_graph_node will store: { (orig_junction_id, wire_id): graph_node_id }
    # Esto mapea un ID de unión original y uno de los dos nanohilos que se cruzan,
    # a un ID de nodo de grafo único.
    wire_junction_to_graph_node = {}

    # Crear los dos nodos para cada unión original y la arista del memristor
    for j_data in junctions:
        orig_junction_id = j_data['id']
        w1_id, w2_id = j_data['wires'] # Los dos nanohilos involucrados en esta unión

        # Asignar IDs de nodo únicos: impar para el primer nanohilo, par para el segundo
        node_for_w1_side = 2 * orig_junction_id + 1  # Nodo impar para el primer nanohilo de la tupla
        node_for_w2_side = 2 * orig_junction_id      # Nodo par para el segundo nanohilo de la tupla

        # Guardar el mapeo para usarlo más tarde al conectar segmentos
        wire_junction_to_graph_node[(orig_junction_id, w1_id)] = node_for_w1_side
        wire_junction_to_graph_node[(orig_junction_id, w2_id)] = node_for_w2_side

        # Añadir estos dos nodos al grafo con su posición
        G.add_node(node_for_w1_side, pos=j_data['pos'])
        G.add_node(node_for_w2_side, pos=j_data['pos'])

        # Añadir la arista del memristor entre estos dos nodos
        # Inicialmente en estado G_OFF (Leído desde el diccionario p)
        G.add_edge(node_for_w1_side, node_for_w2_side, is_memristor=True, conductance=p['G_OFF'])

    # Añadir los segmentos de nanohilos (resistencias)
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
            u_orig_junction_data = junctions_with_dist[k][1] # Primera unión en el segmento
            v_orig_junction_data = junctions_with_dist[k+1][1] # Segunda unión en el segmento
            dist_between = junctions_with_dist[k+1][0] - junctions_with_dist[k][0]

            # Obtener los IDs de nodo de grafo correctos para este 'wire_id' en estas uniones
            u_graph_node = wire_junction_to_graph_node[(u_orig_junction_data['id'], wire_id)]
            v_graph_node = wire_junction_to_graph_node[(v_orig_junction_data['id'], wire_id)]

            G.add_edge(u_graph_node, v_graph_node, weight=dist_between)

    simulation["graph"] = G
    return


# ==============================================================================
# DETECCIÓN DE ELECTRODOS
# ==============================================================================
def find_electrode_nodes2(simulation:dict):
    ''' Encuentra los nodos conectados a los electrodos.
        Extrae el tamaño del sustrato (AREA) y el umbral de borde (PROXIMITY_THRESHOLD)
        directamente de los parámetros centralizados en config.ini.
    '''
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

    simulation["terminals"]={
        "input_nodes": input_nodes,                
        "output_nodes": output_nodes
    }
    return


# ==============================================================================
# FUNCIONES DE PERCOLACIÓN Y CAMINOS
# ==============================================================================
def check_percolation(simulation:dict):
    """ Verifica si existe al menos un camino que conecte la entrada con la salida """

    G=simulation["G"]
    input_nodes = simulation["input_nodes"]
    output_nodes = simulation["output_nodes"]

    for in_node in input_nodes:
        for out_node in output_nodes:
            if nx.has_path(G, in_node, out_node):
                return True
    return False

def get_shortest_path_length(simulation:dict):
    """ Calcula la longitud topológica del camino más corto que conecta los extremos """
    G=simulation["G"]
    input_nodes = simulation["input_nodes"]
    output_nodes = simulation["output_nodes"]
    
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
