# fisica.py
import numpy as np
from scipy.sparse import lil_matrix



# ==============================================================================
# CONSTRUCCIÓN DE LA MATRIZ DE ADMITANCIA (SPARSE)
# ==============================================================================
def build_admittance_matrix2(simulation:dict):
    '''
    Construye la matriz de admitancia del circuito usando representaciones dispersas (CSR).
    Toma los voltajes y constantes directamente desde el archivo de configuración.
    '''

    p = simulation["parameters"]
    G=simulation["graph"]
    terminals = simulation["terminals"]
    input_nodes = terminals["input_nodes"]
    output_nodes = terminals["output_nodes"]
    
    N = G.number_of_nodes()
    node_to_index = {node: i for i, node in enumerate(G.nodes)}

    # Usamos lil_matrix para el llenado eficiente
    Y = lil_matrix((N, N))
    I_vec = np.zeros(N)

    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_to_index[u], node_to_index[v]
        conductance = 0.0

        if data.get('is_memristor', False):
            # Lee G_OFF desde el diccionario
            conductance = data.get('conductance', p['G_OFF'])
        else:
            # Lee R_WIRE_PER_LENGTH calculado con pi desde el diccionario
            conductance = 1.0 / (data['weight'] * p['R_WIRE_PER_LENGTH'] + 1e-12)

        Y[u_idx, v_idx] -= conductance
        Y[v_idx, u_idx] -= conductance
        Y[u_idx, u_idx] += conductance
        Y[v_idx, v_idx] += conductance

    # Condiciones de contorno para los electrodos de entrada y salida
    for node_id in input_nodes:
        idx = node_to_index[node_id]
        Y[idx, :] = 0.0; Y[idx, idx] = 1.0; I_vec[idx] = p['V_INPUT']
        
    for node_id in output_nodes:
        idx = node_to_index[node_id]
        Y[idx, :] = 0.0; Y[idx, idx] = 1.0; I_vec[idx] = p['V_GROUND']

    # --- ESTABILIZACIÓN: Fuga a tierra mínima para evitar singularidad ---
    G_LEAK = 1e-12  # Conductancia de fuga muy pequeña (1 pS)

    for i in range(N):
        Y[i, i] += G_LEAK
        
    # Convertimos a CSR (Compressed Sparse Row) para que el solver vuele
    circuit_data = {
        "Y":Y.tocsr(),
        "I": I_vec,
        "node_to_index": node_to_index,
    }
    simulation["circuit"] = circuit_data
    return circuit_data["Y"], circuit_data["I"], circuit_data["node_to_index"]


# ==============================================================================
# ACTUALIZACIÓN ESTOCÁSTICA DE MEMRISTORES
# ==============================================================================
def update_stochastic_conductance2(G, V_solved, node_to_index):
    '''
    Aplica la dinámica estocástica de conmutación volátil y disolución de filamentos
    leyendo las tasas y sensibilidades desde config.ini.
    '''
    edges_to_set = []
    edges_to_reset = []

    for u, v, data in G.edges(data=True):
        if not data.get('is_memristor', False):
            continue

        u_idx, v_idx = node_to_index[u], node_to_index[v]
        V_mem = abs(V_solved[u_idx] - V_solved[v_idx])
        current_G = data.get('conductance', p['G_OFF'])

        # --- LÓGICA DE SET (Facilitación) ---
        if current_G == p['G_OFF']:
            if V_mem > p['V_THRESHOLD']:
                # Probabilidad exponencial basada en la activación iónica (Ag+)
                p_set = p['P0_SET'] * np.exp(p['ALPHA_SET'] * np.abs(V_mem - p['V_THRESHOLD']))
                if np.random.rand() < p_set:
                    edges_to_set.append((u, v))

        # --- LÓGICA DE RESET (Relajación Volátil - Fig 2b) ---
        elif current_G == p['G_ON']:
            # Factor de estabilidad: disminuye el decay si hay voltaje suficiente
            estabilidad = np.exp(-np.abs(V_mem) / p['V_THRESHOLD'])
            p_decay = p['P_DECAY'] * estabilidad

            if np.random.rand() < p_decay:
                edges_to_reset.append((u, v))

    # Aplicar cambios en lote
    for u, v in edges_to_set: G.edges[u, v]['conductance'] = p['G_ON']
    for u, v in edges_to_reset: G.edges[u, v]['conductance'] = p['G_OFF']

    return G


# ==============================================================================
# CÁLCULOS DE CORRIENTES
# ==============================================================================
def calculate_output_current(G, V_solved, node_to_index, output_nodes):
    total_current = 0.0
    for out_node in output_nodes:
        V_out = V_solved[node_to_index[out_node]]
        for neighbor in G.neighbors(out_node):
            if neighbor not in output_nodes:
                edge_data = G.get_edge_data(out_node, neighbor)
                G_segment = 1.0 / (edge_data['weight'] * p['R_WIRE_PER_LENGTH'] + 1e-9)
                V_neighbor = V_solved[node_to_index[neighbor]]
                total_current += (V_neighbor - V_out) * G_segment
    return total_current

def calculate_input_current(G, V_solved, node_to_index, input_nodes):
    """ Calcula la corriente total que sale de los nodos de entrada (V_INPUT). """
    total_current_from_input = 0.0
    for in_node in input_nodes:
        V_in = V_solved[node_to_index[in_node]]
        for neighbor in G.neighbors(in_node):
            edge_data = G.get_edge_data(in_node, neighbor)
            conductance = 0.0
            
            if edge_data.get('is_memristor', False):
                conductance = edge_data.get('conductance', p['G_OFF'])
            else:
                conductance = 1.0 / (edge_data['weight'] * p['R_WIRE_PER_LENGTH'] + 1e-9)

            V_neighbor = V_solved[node_to_index[neighbor]]
            total_current_from_input += (V_in - V_neighbor) * conductance
    return total_current_from_input


# ==============================================================================
# SEÑALES DE ENTRADA EXTRA
# ==============================================================================
def get_v_ramp(t, total_time, amplitude):
    """ Genera una señal senoidal para ver el ciclo de histéresis """
    return amplitude * np.sin(2 * np.pi * t / total_time)
