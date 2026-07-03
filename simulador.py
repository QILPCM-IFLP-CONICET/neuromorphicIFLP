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
#==============================================================================
#
#  Simulación de pulsos
#
#=============================================================================

def run_simulation_dynamic_pulse(num_wires_to_simulate):
    print(f"--- Simulación de Red de Nanohilos N = {num_wires_to_simulate} (Pulso y Relajación - Fig 2b) ---")
    wires, junctions, wire_map = geo.generate_and_find_junctions(num_wires_to_simulate, LENGTH, AREA)

    G = grf.build_graph2(wires, junctions, wire_map)
    input_nodes, output_nodes = grf.find_electrode_nodes2(G, AREA, PROXIMITY_THRESHOLD)

    if len(input_nodes) == 0 or len(output_nodes) == 0:
        print("Error: Red no conectada a electrodos."); return

    N_orig_junctions = len(junctions)

    print(f"Número de memristores: {N_orig_junctions} -> Nodos de Grafo: {G.number_of_nodes()} | Inputs: {len(input_nodes)} | Outputs: {len(output_nodes)}")
    #print(f"Areal density: {num_wires_to_simulate/(AREA*AREA)} ")


    # Estado inicial: Todos los memristores en G_OFF
    for u, v, data in G.edges(data=True):
        if data.get('is_memristor', False):
            G.edges[u,v]['conductance'] = G_OFF

    history_time = []
    history_G_total = []
    history_active = []
    current_time = 0.0

    # Configuración del experimento (basado en Fig 2b)
    T_PULSE = 10.0        # 10 segundos de pulso alto [cite: 678]
    T_RELAX = 40.0       # Tiempo de relajación
    V_PULSE = 4.0         # Voltaje de facilitación (ejemplo de la Fig 2b)
    V_READ = 0.05         # 50 mV para lectura

    total_steps = int((T_PULSE + T_RELAX) / TIME_STEP_DT)

    print(f"Iniciando: {T_PULSE}s pulso ({V_PULSE}V) + {T_RELAX}s relajación ({V_READ}V)")

    for step in range(total_steps):
        # Determinamos el voltaje según el tiempo actual
        if current_time <= T_PULSE:
            v_now = V_PULSE
        else:
            v_now = V_READ

        # 1. Resolver Circuito con el voltaje dinámico v_now
        Y, I_vec, n2i = fis.build_admittance_matrix2(G, v_now, V_GROUND, input_nodes, output_nodes, R_WIRE_PER_LENGTH, G_OFF)
        try:
            V_vec = spsolve(Y, I_vec)
        except:
            print("Error en matriz de admitancia."); break

        # 2. Calcular Conductancia Total (G = I_in / V_actual)
        I_in = fis.calculate_input_current(G, V_vec, n2i, input_nodes, R_WIRE_PER_LENGTH)
        # Usamos v_now para la conductancia
        G_total = 1000 * (I_in / v_now) if v_now != 0 else 0

        count_ON = sum(1 for u, v, data in G.edges(data=True)
                      if data.get('is_memristor', False) and data.get('conductance') == G_ON)

        history_time.append(current_time)
        history_G_total.append(G_total)
        history_active.append(count_ON)

        # 3. Actualización Estocástica (Difusión y Disolución de filamentos)
        # Durante la relajación (V_READ), dominará el pdecay (volatilidad) [cite: 682]
        #pcero = P0_SET * TIME_STEP_DT
        #pdecay = P_DECAY * TIME_STEP_DT
        pcero = P0_SET
        pdecay = P_DECAY
        G = fis.update_stochastic_conductance2(G, V_vec, n2i, G_ON, G_OFF, V_THRESHOLD, pcero, ALPHA_SET, pdecay)

        current_time += TIME_STEP_DT

        if step % int(total_steps/10) == 0:
            mode = "PULSO" if current_time <= T_PULSE else "RELAX"
            print(f"[{mode}] T={current_time:.1f}s | G={G_total} mS | ON={count_ON}")

    return history_time, history_G_total, history_active
