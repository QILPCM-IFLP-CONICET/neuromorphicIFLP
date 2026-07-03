import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Importamos nuestros módulos hermanos
import geometria as geo
import grafo as grf
import fisica as fis

TOTAL_TIME = 0.5
TIME_STEP_DT = 1e-3
V_INPUT = 3.6
V_GROUND = 0.0



AREA = 1000.0  #  define un substrato de Area*Area en micras^2 (ya se, mal elegido el nombre)
LENGTH = 70.0  # longitud de los nanohilos en micras, por ahora son todos iguales

PROXIMITY_THRESHOLD = 0.05*AREA

def run_simulation_dynamic2(num_wires_to_simulate):
    print("--- Simulación de Red de Nanohilos (Conductancia Dinámica) ---")
    # CORREGIDO: Ahora llamamos usando el alias 'geo'
    wires, junctions, wire_map = geo.generate_and_find_junctions(num_wires_to_simulate)
    
    # CORREGIDO: Ahora llamamos usando el alias 'grf'
    G = grf.build_graph2(wires, junctions, wire_map)
    input_nodes, output_nodes = grf.find_electrode_nodes2(G, AREA, PROXIMITY_THRESHOLD)

    if not grf.check_percolation(G, input_nodes, output_nodes):
        print("Error: La red generada no percoló numéricamente. Incrementá hilos.")
        return None

    history_time, history_G_total, history_active = [], [], []
    current_time = 0.0
    steps = int(TOTAL_TIME / TIME_STEP_DT)

    for step in range(steps):
        # CORREGIDO: Ahora llamamos usando el alias 'fis'
        Y, I_vec, n2i = fis.build_admittance_matrix2(G, V_INPUT, V_GROUND, input_nodes, output_nodes)
        try:
            V_vec = spsolve(Y, I_vec)
        except Exception as e:
            print(f"Error numérico en paso {step}: {e}"); break

        I_in = fis.calculate_input_current(G, V_vec, n2i, input_nodes)
        G_total = 1000 * (I_in / V_INPUT) 
        count_ON = sum(1 for u, v, d in G.edges(data=True) if d.get('is_memristor', False) and d.get('conductance') == fis.G_ON)

        history_time.append(current_time)
        history_G_total.append(G_total)
        history_active.append(count_ON)

        pcero = 0.4 * TIME_STEP_DT
        pdecay = 0.8 * TIME_STEP_DT
        # CORREGIDO: Ahora llamamos usando el alias 'fis'
        G = fis.update_stochastic_conductance2(G, V_vec, n2i, fis.G_ON, fis.G_OFF, fis.V_THRESHOLD, pcero, 1.0, pdecay)
        current_time += TIME_STEP_DT

    return history_time, history_G_total, history_active

def run_simulation_dynamic_histeresis(num_wires_to_simulate, cantidad_ciclos=3):
    print("--- Simulación de Red de Nanohilos (Ciclo de Histéresis) ---")
    # CORREGIDO: Uso de alias correspondientes
    wires, junctions, wire_map = geo.generate_and_find_junctions(num_wires_to_simulate)
    G = grf.build_graph2(wires, junctions, wire_map)
    input_nodes, output_nodes = grf.find_electrode_nodes2(G)

    history_time, history_v_in, history_i_in, history_active = [], [], [], []
    steps = int(TOTAL_TIME / TIME_STEP_DT)

    for step in range(steps):
        current_time = step * TIME_STEP_DT
        v_now = V_INPUT * np.sin(2 * np.pi * cantidad_ciclos * current_time / TOTAL_TIME)

        Y, I_vec, n2i = fis.build_admittance_matrix2(G, v_now, V_GROUND, input_nodes, output_nodes)
        try:
            V_vec = spsolve(Y, I_vec)
        except: break

        I_in = fis.calculate_input_current(G, V_vec, n2i, input_nodes)
        count_ON = sum(1 for u, v, d in G.edges(data=True) if d.get('is_memristor', False) and d.get('conductance') == fis.G_ON)

        history_time.append(current_time)
        history_v_in.append(v_now)
        history_i_in.append(I_in)
        history_active.append(count_ON)

        pcero = 0.4 * TIME_STEP_DT
        pdecay = 0.8 * TIME_STEP_DT
        G = fis.update_stochastic_conductance2(G, V_vec, n2i, fis.G_ON, fis.G_OFF, fis.V_THRESHOLD, pcero, 1.0, pdecay)

    return history_time, history_v_in, history_i_in, history_active