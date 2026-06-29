import numpy as np
from scipy.sparse import lil_matrix

rho_plata = 1.59e-2                  
radioNW = (115/2) * 1e-3             
R_WIRE_PER_LENGTH = rho_plata / (np.pi * radioNW * radioNW)

G_ON = 1.0 / 1e3                     
G_OFF = 1.0 / 1e9                    
V_THRESHOLD = 0.01

def build_admittance_matrix2(G, V_INPUT, V_GROUND, input_nodes, output_nodes, 
                             r_wire_per_length=R_WIRE_PER_LENGTH, g_off_default=G_OFF):
    N = G.number_of_nodes()
    node_to_index = {node: i for i, node in enumerate(G.nodes)}

    Y = lil_matrix((N, N))
    I_vec = np.zeros(N)

    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        conductance = 0.0

        if data.get('is_memristor', False):
            conductance = data.get('conductance', g_off_default)
        else:
            conductance = 1.0 / (data['weight'] * r_wire_per_length + 1e-12)

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

    G_LEAK = 1e-12
    for i in range(N): Y[i, i] += G_LEAK

    return Y.tocsr(), I_vec, node_to_index

def update_stochastic_conductance2(G, V_solved, node_to_index, G_ON_val=G_ON, G_OFF_val=G_OFF, 
                                   V_THRESHOLD_val=V_THRESHOLD, P0_SET_val=0.4, ALPHA_SET_val=1.0, P_DECAY_val=0.8):
    edges_to_set, edges_to_reset = [], []

    for u, v, data in G.edges(data=True):
        if not data.get('is_memristor', False): continue

        u_idx, v_idx = node_to_index[u], node_to_index[v]
        V_mem = abs(V_solved[u_idx] - V_solved[v_idx])
        current_G = data.get('conductance', G_OFF_val)

        if current_G == G_OFF_val:
            if V_mem > V_THRESHOLD_val:
                p_set = P0_SET_val * np.exp(ALPHA_SET_val * np.abs(V_mem - V_THRESHOLD_val))
                if np.random.rand() < p_set: edges_to_set.append((u, v))
        elif current_G == G_ON_val:
            estabilidad = np.exp(-np.abs(V_mem) / V_THRESHOLD_val)
            p_decay = P_DECAY_val * estabilidad
            if np.random.rand() < p_decay: edges_to_reset.append((u, v))

    for u, v in edges_to_set: G.edges[u, v]['conductance'] = G_ON_val
    for u, v in edges_to_reset: G.edges[u, v]['conductance'] = G_OFF_val
    return G

def calculate_input_current(G, V_solved, node_to_index, input_nodes, r_wire_per_length=R_WIRE_PER_LENGTH):
    total_current_from_input = 0.0
    for in_node in input_nodes:
        V_in = V_solved[node_to_index[in_node]]
        for neighbor in G.neighbors(in_node):
            edge_data = G.get_edge_data(in_node, neighbor)
            conductance = edge_data.get('conductance', G_OFF) if edge_data.get('is_memristor', False) else 1.0 / (edge_data['weight'] * r_wire_per_length + 1e-9)
            V_neighbor = V_solved[node_to_index[neighbor]]
            total_current_from_input += (V_in - V_neighbor) * conductance
    return total_current_from_input