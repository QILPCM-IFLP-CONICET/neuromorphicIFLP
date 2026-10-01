# simulador.py
import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

# Importamos nuestros módulos hermanos
from . import fisica as fis



# ==============================================================================
# SIMULACIÓN DE PULSOS DINÁMICOS
# ==============================================================================
def run_simulation_dynamic_pulse(simulation:dict):
    '''
    Ejecuta la simulación dinámica de pulso y relajación (Fig 2b del paper).
    Toma las constantes de tamaño de red, tiempos, conductancias y probabilidades
    directamente desde el archivo centralizado config.ini.
    '''
    p = simulation["parameters"]
    num_wires_to_simulate = p['NUM_WIRES']
    
    print(f"--- Simulación de Red de Nanohilos N = {num_wires_to_simulate} (Pulso y Relajación - Fig 2b) ---")
    
    # 1. Generación de la geometría (Firmas limpias sin parámetros redundantes)
    wire_junctions = simulation["junctions"]
    wires, junctions, wire_map = wire_junctions["wires"], wire_junctions["junctions"], wire_junctions["wire_to_jnctions"]
    G = simulation["graph"]
    terminals = simulation["terminals"]
    input_nodes, output_nodes = terminals["input_nodes"], terminals["output_nodes"]

    if len(input_nodes) == 0 or len(output_nodes) == 0:
        print("Error: Red no conectada a electrodos.")
        return

    N_orig_junctions = len(junctions)
    print(f"Número de memristores: {N_orig_junctions} -> Nodos de Grafo: {G.number_of_nodes()} | Inputs: {len(input_nodes)} | Outputs: {len(output_nodes)}")

    # Estado inicial explícito usando p['G_OFF']
    for u, v, data in G.edges(data=True):
        if data.get('is_memristor', False):
            G.edges[u, v]['conductance'] = p['G_OFF']

    history_time = []
    history_G_total = []
    history_active = []
    current_time = 0.0

    # Configuración del experimento temporal (basado en Fig 2b)
    T_PULSE = 10.0       # 10 segundos de pulso alto
    T_RELAX = 40.0       # Tiempo de relajación
    V_PULSE = p['V_INPUT']  # Tomamos el voltaje de entrada como el pulso alto configurado
    V_READ = 0.05        # 50 mV para lectura
    
    dt = p['TIME_STEP_DT']
    total_steps = int((T_PULSE + T_RELAX) / dt)

    print(f"Iniciando: {T_PULSE}s pulso ({V_PULSE}V) + {T_RELAX}s relajación ({V_READ}V)")

    # 3. Bucle Temporal Principal
    for step in range(total_steps):
        # Determinamos el voltaje de entrada según el tiempo actual
        if current_time <= T_PULSE:
            v_now = V_PULSE
        else:
            v_now = V_READ

        # Modificamos dinámicamente V_INPUT en el diccionario temporal para esta iteración
        p['V_INPUT'] = v_now

        # 3.1. Resolver Circuito con el voltaje dinámico v_now
        Y, I_vec, n2i = fis.build_admittance_matrix2(simulation)
        try:
            V_vec = spsolve(Y, I_vec)
        except Exception as e:
            print(f"Error en matriz de admitancia en t={current_time:.3f} s: {e}")
            break

        # 3.2. Calcular Conductancia Total (G = I_in / V_actual)
        I_in = fis.calculate_input_current(G, V_vec, n2i, input_nodes)
        G_total = 1000 * (I_in / v_now) if v_now != 0 else 0

        # Contar cuántos memristores están en estado G_ON robusto
        count_ON = sum(1 for u, v, data in G.edges(data=True)
                       if data.get('is_memristor', False) and data.get('conductance') == p['G_ON'])

        history_time.append(current_time)
        history_G_total.append(G_total)
        history_active.append(count_ON)

        # 3.3. Actualización Estocástica (Difusión y Disolución de filamentos)
        G = fis.update_stochastic_conductance2(G, V_vec, n2i)

        current_time += dt

        # Imprimir logs de control cada 10% del proceso
        if step % max(1, int(total_steps / 10)) == 0:
            mode = "PULSO" if current_time <= T_PULSE else "RELAX"
            print(f"[{mode}] T={current_time:.1f}s | G={G_total:.4e} mS | ON={count_ON}")

    return history_time, history_G_total, history_active
