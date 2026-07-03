import numpy as np
from scipy.sparse import lil_matrix

rho_plata = 1.59e-2                  
radioNW = (115/2) * 1e-3             
R_WIRE_PER_LENGTH = rho_plata / (np.pi * radioNW * radioNW)

G_ON = 1.0 / 1e3                     
G_OFF = 1.0 / 1e9                    
V_THRESHOLD = 0.01


#--------------------
#Nueva forma de armar la matriz con matrices sparse y una fuga de corriente
#--------------------

def build_admittance_matrix2(G, V_INPUT, V_GROUND, input_nodes, output_nodes, R_WIRE_PER_LENGTH, G_OFF_default):
    N = G.number_of_nodes()
    node_to_index = {node: i for i, node in enumerate(G.nodes)}

    # CAMBIO: Usamos lil_matrix en lugar de np.zeros
    Y = lil_matrix((N, N))
    I_vec = np.zeros(N)

    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        conductance = 0.0

        if data.get('is_memristor', False):
            conductance = data.get('conductance', G_OFF_default)
        else:
            conductance = 1.0 / (data['weight'] * R_WIRE_PER_LENGTH + 1e-12)

        # La sintaxis de llenado es idéntica a la anterior
        Y[u_idx, v_idx] -= conductance
        Y[v_idx, u_idx] -= conductance
        Y[u_idx, u_idx] += conductance
        Y[v_idx, v_idx] += conductance

    for node_id in input_nodes:
        idx = node_to_index[node_id]
        Y[idx, :] = 0.0; Y[idx, idx] = 1.0; I_vec[idx] = V_INPUT
    for node_id in output_nodes:
        idx = node_to_index[node_id]
        Y[idx, :] = 0.0; Y[idx, idx] = 1.0; I_vec[idx] = V_GROUND

    # --- ESTABILIZACIÓN: Fuga a tierra mínima para evitar singularidad ---
    G_LEAK = 1e-12  # Conductancia de fuga muy pequeña (1 pS)

    for i in range(N):
        Y[i, i] += G_LEAK
    # CAMBIO: Convertimos a CSR (Compressed Sparse Row) para que el solver vuele
    return Y.tocsr(), I_vec, node_to_index


##############################################################################
#
# ACA ESTÁ LA ACTUALIZACIÓN ESTOCÁSTICA
#
#############################################################################



def update_stochastic_conductance2(G, V_solved, node_to_index, G_ON_val, G_OFF_val, V_THRESHOLD_val, P0_SET_val, ALPHA_SET_val, P_DECAY_val):
    edges_to_set = []
    edges_to_reset = []

    for u, v, data in G.edges(data=True):
        if not data.get('is_memristor', False):
            continue

        u_idx, v_idx = node_to_index[u], node_to_index[v]
        V_mem = abs(V_solved[u_idx] - V_solved[v_idx])
        current_G = data.get('conductance', G_OFF_val)

        # --- LÓGICA DE SET (Facilitación) ---
        if current_G == G_OFF_val:
            if V_mem > V_THRESHOLD_val:
                # Probabilidad exponencial basada en la activación iónica (Ag+)
                # dt ya debe estar considerado en P0_SET_val (prob * dt)
                p_set = P0_SET_val * np.exp(ALPHA_SET_val * np.abs(V_mem - V_THRESHOLD_val))
                if np.random.rand() < p_set:
                    edges_to_set.append((u, v))

        # --- LÓGICA DE RESET (Relajación Volátil - Fig 2b) ---
        elif current_G == G_ON_val:
            # En el paper, el filamento es inestable.
            # Si V_mem es bajo (como el V_read de 50mV), el filamento se disuelve.
            # Si V_mem es alto, el campo eléctrico lo mantiene estable.

            # Factor de estabilidad: disminuye el decay si hay voltaje suficiente
            estabilidad = np.exp(-np.abs(V_mem) / V_THRESHOLD_val)
            p_decay = P_DECAY_val * estabilidad

            if np.random.rand() < p_decay:
                edges_to_reset.append((u, v))

    # Aplicar cambios
    for u, v in edges_to_set: G.edges[u, v]['conductance'] = G_ON_val
    for u, v in edges_to_reset: G.edges[u, v]['conductance'] = G_OFF_val

    return G


def calculate_output_current(G, V_solved, node_to_index, output_nodes):
    total_current = 0.0
    for out_node in output_nodes:
        V_out = V_solved[node_to_index[out_node]]
        for neighbor in G.neighbors(out_node):
            if neighbor not in output_nodes:
                edge_data = G.get_edge_data(out_node, neighbor)
                G_segment = 1.0 / (edge_data['weight'] * R_WIRE_PER_LENGTH + 1e-9)
                V_neighbor = V_solved[node_to_index[neighbor]]
                total_current += (V_neighbor - V_out) * G_segment
    return total_current

def calculate_input_current(G, V_solved, node_to_index, input_nodes, R_WIRE_PER_LENGTH):
    """ Calcula la corriente total que sale de los nodos de entrada (V_INPUT).
        Para la nueva estructura de grafo, sumamos la corriente que fluye
        desde cada nodo de entrada a sus vecinos (a través de los memristores).
    """
    total_current_from_input = 0.0
    for in_node in input_nodes: # in_node es un 'source_node_id'
        V_in = V_solved[node_to_index[in_node]] # Este debería ser V_INPUT
        for neighbor in G.neighbors(in_node):
            edge_data = G.get_edge_data(in_node, neighbor)
            conductance = 0.0
            # En la nueva topología, los input_nodes (source_node_id) solo están conectados a su sink_node_id via memristor.
            # Por lo tanto, esta arista siempre debería ser un memristor.
            if edge_data.get('is_memristor', False):
                conductance = edge_data.get('conductance', G_OFF)
            else: # Esto no debería ocurrir para los nodos de entrada en la estructura actual
                conductance = 1.0 / (edge_data['weight'] * R_WIRE_PER_LENGTH + 1e-9)

            V_neighbor = V_solved[node_to_index[neighbor]]
            # Corriente que fluye DESDE in_node HACIA neighbor
            total_current_from_input += (V_in - V_neighbor) * conductance
    return total_current_from_input



def get_v_ramp(t, total_time, amplitude):
    """ Genera una señal senoidal para ver el ciclo de histéresis """
    return amplitude * np.sin(2 * np.pi * t / total_time)

