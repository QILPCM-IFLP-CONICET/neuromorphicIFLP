# visualizacion.py
import matplotlib.pyplot as plt
import numpy as np


def plot_simulation_results(simulation, history_time, history_g_total):
    """
    Grafica la evolución temporal de la conductancia de la red durante un
    experimento de pulso y relajación (Fig 2b).

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación (para leer parámetros del experimento).
    history_time : list or np.ndarray
        Vector de tiempos (s).
    history_g_total : list or np.ndarray
        Vector de conductancia equivalente total (mS).
    """
    p = simulation["parameters"]
    t = np.array(history_time)
    graph = np.array(history_g_total)

    if t.size == 0:
        print("⚠️ No hay datos para graficar (historial vacío).")
        return

    t_pulse = p["T_PULSE"]
    v_pulse = p["V_INPUT"]
    v_read = p["V_READ"]

    plt.figure(figsize=(10, 6))
    plt.plot(t, graph, color="#2c3e50", linewidth=2, label="Conductancia de la Red")

    plt.axvspan(0, t_pulse, color="yellow", alpha=0.2, label=f"Pulso ({t_pulse:.0f}s @ {v_pulse}V)")
    plt.axvspan(t_pulse, t[-1], color="gray", alpha=0.1, label=f"Relajación ({v_read}V)")

    plt.title("Dinámica de Conductancia: Fase de Facilitación y Relajación", fontsize=14)
    plt.xlabel("Tiempo (s)", fontsize=12)
    plt.ylabel("Conductancia (mS)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()

    ymax = np.max(graph) if graph.size > 0 else 1.0
    mid_pulse = t_pulse / 2
    mid_relax = t_pulse + (t[-1] - t_pulse) / 2

    plt.annotate(
        "Potenciación\n(Facilitation)",
        xy=(mid_pulse, ymax * 0.5),
        ha="center",
        fontweight="bold",
        color="orange",
    )
    plt.annotate(
        "Decaimiento\n(Relaxation)",
        xy=(mid_relax, ymax * 0.2),
        ha="center",
        fontweight="bold",
        color="blue",
    )

    plt.tight_layout()
    plt.show()
