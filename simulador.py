import numpy as np
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

import geometria
import grafo
import fisica


TOTAL_TIME = 0.5
TIME_STEP_DT = 1e-3
V_INPUT = 3.6
V_GROUND = 0.0

# ==============================================================================
# 4. EJECUCIÓN
# ==============================================================================

def run_simulation_dynamic2(num_wires_to_simulate):
    print("--- Simulación de Red de Nanohilos (Conductancia Dinámica - Memristores como Aristas) ---")
    wires, junctions, wire_map = generate_and_find_junctions(num_wires_to_simulate, LENGTH, AREA)
    N_orig_junctions = len(junctions)

    G = build_graph2(wires, junctions, wire_map)
    input_nodes, output_nodes = find_electrode_nodes2(G, AREA, PROXIMITY_THRESHOLD) # Eliminado N_orig_junctions de la llamada

    if len(input_nodes) == 0 or len(output_nodes) == 0:
        print("Error: Red no conectada a electrodos. Reintenta.")
        return

    print(f"Número de memristores: {N_orig_junctions} -> Nodos de Grafo: {G.number_of_nodes()} | Inputs: {len(input_nodes)} | Outputs: {len(output_nodes)}")
    print(f"Areal density: {num_wires_to_simulate/(AREA*AREA)} ")

    # Estado inicial de los memristores (aristas)
    for u, v, data in G.edges(data=True):
        if data.get('is_memristor', False):
            G.edges[u,v]['conductance'] = G_OFF

    history_time = []
    history_G_total = []
    history_active = []
    current_time = 0.0

    print("Iniciando bucle temporal...")
    steps = int(TOTAL_TIME / TIME_STEP_DT)

    for step in range(steps):
        # 1. Resolver Circuito
        Y, I_vec, n2i = build_admittance_matrix2(G, V_INPUT, V_GROUND, input_nodes, output_nodes, R_WIRE_PER_LENGTH, G_OFF)
        try:
            #V_vec = solve(Y, I_vec)
            V_vec = spsolve(Y, I_vec)
        except np.linalg.LinAlgError:
            print("Error de singularidad en la matriz de admitancia. Posiblemente un grafo desconectado."); break

        # 2. Actualizar voltajes internos (en este modelo, los voltajes se almacenan en V_vec, no en los nodos de G)

        # 3. Calcular Conductancia Total (G = I_in / V_INPUT)
        I_in = calculate_input_current(G, V_vec, n2i, input_nodes, R_WIRE_PER_LENGTH)
        G_total = 1000*(I_in / V_INPUT)

        count_ON = sum(1 for u, v, data in G.edges(data=True) if data.get('is_memristor', False) and data.get('conductance', G_OFF) == G_ON)

        history_time.append(current_time)
        history_G_total.append(G_total)
        history_active.append(count_ON)

        # 4. Actualización Estocástica de memristores (aristas)
        pcero = P0_SET * TIME_STEP_DT
        pdecay = P_DECAY * TIME_STEP_DT
        G = update_stochastic_conductance2(G, V_vec, n2i, G_ON, G_OFF, V_THRESHOLD, pcero, ALPHA_SET, pdecay)

        current_time += TIME_STEP_DT

        if step % 100 == 0:
            print(f"Step {step}/{steps} | T={current_time:.3f}s | G_tot={G_total:.2e} mS | Activos={count_ON}")


    return history_time, history_G_total, history_active


#--------------------------------------------------------------
#  CICLO DE HISTERESIS
#--------------------------------------------------------------

def run_simulation_dynamic_histeresis(num_wires_to_simulate):
    print("--- Simulación de Red de Nanohilos (Sparse + Histéresis) ---")

    # 1. Generación de la red (Igual que antes)
    wires, junctions, wire_map = generate_and_find_junctions(num_wires_to_simulate, LENGTH, AREA)
    N_orig_junctions = len(junctions)
    G = build_graph2(wires, junctions, wire_map)
    input_nodes, output_nodes = find_electrode_nodes2(G, AREA, PROXIMITY_THRESHOLD)

    if len(input_nodes) == 0 or len(output_nodes) == 0:
        print("Error: Red no conectada a electrodos. Reintenta.")
        return

    print(f"Número de memristores: {N_orig_junctions} -> Nodos de Grafo: {G.number_of_nodes()} | Inputs: {len(input_nodes)} | Outputs: {len(output_nodes)}")
    print(f"Areal density: {num_wires_to_simulate/(AREA*AREA)} ")



    # Listas para el historial
    history_time = []
    history_v_in = []
    history_i_in = []
    history_active = []

    steps = int(TOTAL_TIME / TIME_STEP_DT)

    print(f"Iniciando simulación con {num_wires_to_simulate} nanohilos...")

    for step in range(steps):
        current_time = step * TIME_STEP_DT

        # --- CAMBIO 1: Rampa de Voltaje (Senoidal) ---
        # Esto genera el ciclo 0 -> V_max -> 0 -> -V_max -> 0

        catidad_ciclos = 3 # <<<<<<<<<<<

        v_now = V_INPUT * np.sin(2 * np.pi * catidad_ciclos * current_time / TOTAL_TIME) ### <<< sele puede agregar un factor para hacer varios ciclos de histeresis

        # --- CAMBIO 2: Matriz Sparse con G_LEAK ---
        # llamamos a LA función modificada con lil_matrix y tocsr()
        Y, I_vec, n2i = build_admittance_matrix2(G, v_now, V_GROUND, input_nodes, output_nodes, R_WIRE_PER_LENGTH, G_OFF)

        # --- CAMBIO 3: Solver Sparse ---
        try:
            V_vec = spsolve(Y, I_vec)
        except Exception as e:
            print(f"Error en el solver: {e}")
            break

        # --- CAMBIO 4: Registro de datos para Histéresis ---
        I_in = calculate_input_current(G, V_vec, n2i, input_nodes, R_WIRE_PER_LENGTH)
        count_ON = sum(1 for u, v, data in G.edges(data=True) if data.get('is_memristor', False) and data.get('conductance', G_OFF) == G_ON)

        history_time.append(current_time)
        history_v_in.append(v_now)
        history_i_in.append(I_in)
        history_active.append(count_ON)

        # 5. Actualización Estocástica pero escaleando la probabilidad con el tiempo entre medidas de I
        pcero = P0_SET * TIME_STEP_DT
        pdecay = P_DECAY * TIME_STEP_DT

        G = update_stochastic_conductance2(G, V_vec, n2i, G_ON, G_OFF, V_THRESHOLD, pcero, ALPHA_SET, pdecay)

        if step % 100 == 0:
            print(f"Step {step}/{steps} | V={v_now:.2f}V | I={I_in:.2e}A | Activos={count_ON}")


    return history_time, history_v_in, history_i_in, history_active

#--------------------------------------------------------------

# ==============================================================================
# 5. FUNCIONES DE AUTOMATIZACIÓN Y ANÁLISIS
# ==============================================================================

def generar_datos_histeresis_variando_nw(lista_num_wires, folder_path="/content/drive/MyDrive/1-Investigacion/2026_ComputacionNeuromorfica/Simulacion2026/datosHisteresis/V3p6"):
    """
    Ejecuta simulaciones para diferentes cantidades de nanohilos y guarda V e I.
    """
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Carpeta creada: {folder_path}")

    for nw in lista_num_wires:
        print(f"\n>>> Simulando para NW = {nw}...")

        # Ejecutamos la simulación de histéresis
        # Nota: Asegúrate de que TOTAL_TIME sea suficiente para completar un ciclo (Seno)
        time, v_in, i_in, active = run_simulation_dynamic_histeresis(nw)

        # Estructuramos los datos para guardar
        data_to_save = {
            'v': np.array(v_in),
            'i': np.array(i_in),
            'nw': nw
        }

        filename = os.path.join(folder_path, f"histeresis_NW_{nw}.npy")
        np.save(filename, data_to_save)
        print(f"Datos guardados en: {filename}")


#==============================================================================
#
#  Simulación de pulsos
#
#=============================================================================

def run_simulation_dynamic_pulse(num_wires_to_simulate):
    print(f"--- Simulación de Red de Nanohilos N = {num_wires_to_simulate} (Pulso y Relajación - Fig 2b) ---")
    wires, junctions, wire_map = generate_and_find_junctions(num_wires_to_simulate, LENGTH, AREA)

    G = build_graph2(wires, junctions, wire_map)
    input_nodes, output_nodes = find_electrode_nodes2(G, AREA, PROXIMITY_THRESHOLD)

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
        Y, I_vec, n2i = build_admittance_matrix2(G, v_now, V_GROUND, input_nodes, output_nodes, R_WIRE_PER_LENGTH, G_OFF)
        try:
            V_vec = spsolve(Y, I_vec)
        except:
            print("Error en matriz de admitancia."); break

        # 2. Calcular Conductancia Total (G = I_in / V_actual)
        I_in = calculate_input_current(G, V_vec, n2i, input_nodes, R_WIRE_PER_LENGTH)
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
        G = update_stochastic_conductance2(G, V_vec, n2i, G_ON, G_OFF, V_THRESHOLD, pcero, ALPHA_SET, pdecay)

        current_time += TIME_STEP_DT

        if step % int(total_steps/10) == 0:
            mode = "PULSO" if current_time <= T_PULSE else "RELAX"
            print(f"[{mode}] T={current_time:.1f}s | G={G_total} mS | ON={count_ON}")

    return history_time, history_G_total, history_active
