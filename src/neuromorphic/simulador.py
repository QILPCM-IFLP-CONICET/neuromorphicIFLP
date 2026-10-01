# simulador.py
from scipy.sparse.linalg import spsolve

from . import fisica as fis
from .grafo import check_percolation


# ==============================================================================
# SIMULACIÓN DE PULSOS DINÁMICOS
# ==============================================================================
def run_simulation_dynamic_pulse(simulation: dict):
    """
    Ejecuta la simulación dinámica de pulso y relajación (Fig 2b del paper).
    Todos los parámetros se leen desde simulation["parameters"].
    """
    p = simulation["parameters"]
    G = simulation["graph"]
    terminals = simulation["terminals"]
    input_nodes = terminals["input_nodes"]
    output_nodes = terminals["output_nodes"]

    if len(input_nodes) == 0 or len(output_nodes) == 0:
        raise RuntimeError(
            "La red no tiene electrodos conectados: "
            f"input_nodes={len(input_nodes)}, output_nodes={len(output_nodes)}"
        )

    # --- Diagnóstico de conectividad ---
    if not check_percolation(simulation):
        # Estimar densidad vs umbral crítico
        L = p["LENGTH"]
        A = p["AREA"]
        N = p["NUM_WIRES"]
        N_c = 5.7 * A / (L**2)
        raise RuntimeError(
            f"La red NO percola: no hay camino topológico entre electrodos.\n"
            f"  Densidad actual : N = {N} (N·L²/A = {N * L**2 / A:.2f})\n"
            f"  Umbral crítico  : N_c ≈ {N_c:.0f} (N·L²/A ≈ 5.7)\n"
            f"  Sugerencia      : aumentar NUM_WIRES > {int(N_c * 1.3)} "
            f"o agrandar LENGTH a > {int((5.7 * A / N) ** 0.5 * 1.3)} µm."
        )

    N_orig_junctions = len(simulation["junctions"]["junctions"])
    print(
        f"--- Simulación de Red de Nanohilos N = {p['NUM_WIRES']} (Pulso y Relajación - Fig 2b) ---"
    )
    print(
        f"Memristores: {N_orig_junctions} | Nodos de grafo: {G.number_of_nodes()} | "
        f"Inputs: {len(input_nodes)} | Outputs: {len(output_nodes)}"
    )

    # Estado inicial explícito de los memristores
    for _, _, data in G.edges(data=True):
        if data.get("is_memristor", False):
            data["conductance"] = p["G_OFF"]

    history_time = []
    history_G_total = []
    history_active = []
    current_time = 0.0

    T_PULSE = p["T_PULSE"]
    T_RELAX = p["T_RELAX"]
    V_PULSE = p["V_INPUT"]
    V_READ = p["V_READ"]
    dt = p["TIME_STEP_DT"]
    total_steps = int((T_PULSE + T_RELAX) / dt)

    print(f"Iniciando: {T_PULSE}s pulso ({V_PULSE}V) + {T_RELAX}s relajación ({V_READ}V)")

    for step in range(total_steps):
        # Voltaje dinámico según la fase (sin mutar p['V_INPUT'])
        v_now = V_PULSE if current_time <= T_PULSE else V_READ

        # 1) Resolver el circuito con el voltaje actual
        Y, I_vec, _ = fis.build_admittance_matrix2(simulation, v_input=v_now)
        try:
            V_vec = spsolve(Y, I_vec)
        except Exception as e:
            print(f"Error en spsolve en t={current_time:.3f} s: {e}")
            break

        # 2) Conductancia equivalente total
        I_in = fis.calculate_input_current(simulation, V_vec)
        G_total = 1000.0 * (I_in / v_now) if v_now != 0 else 0.0

        # 3) Contar memristores en estado ON
        count_ON = sum(
            1
            for u, v, data in G.edges(data=True)
            if data.get("is_memristor", False) and data.get("conductance") == p["G_ON"]
        )

        history_time.append(current_time)
        history_G_total.append(G_total)
        history_active.append(count_ON)

        # 4) Actualización estocástica (muta G in-place)
        fis.update_stochastic_conductance2(simulation, V_vec)

        current_time += dt

        if step % max(1, total_steps // 10) == 0:
            mode = "PULSO" if current_time <= T_PULSE else "RELAX"
            print(f"[{mode}] T={current_time:.1f}s | G={G_total:.4e} mS | ON={count_ON}")

    return history_time, history_G_total, history_active
