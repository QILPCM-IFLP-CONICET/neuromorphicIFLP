# visualizacion.py
import numpy as np
import matplotlib.pyplot as plt


def plot_simulation_results(simulation, history_time, history_G_total):
    '''
    Grafica la evolución temporal de la conductancia de la red durante un
    experimento de pulso y relajación (Fig 2b).

    Parameters
    ----------
    simulation : dict
        Diccionario de simulación (para leer parámetros del experimento).
    history_time : list or np.ndarray
        Vector de tiempos (s).
    history_G_total : list or np.ndarray
        Vector de conductancia equivalente total (mS).
    '''
    p = simulation["parameters"]
    t = np.array(history_time)
    G = np.array(history_G_total)

    if t.size == 0:
        print("⚠️ No hay datos para graficar (historial vacío).")
        return

    T_PULSE = p['T_PULSE']
    V_PULSE = p['V_INPUT']
    V_READ = p['V_READ']

    plt.figure(figsize=(10, 6))
    plt.plot(t, G, color='#2c3e50', linewidth=2, label='Conductancia de la Red')

    plt.axvspan(0, T_PULSE, color='yellow', alpha=0.2,
                label=f'Pulso ({T_PULSE:.0f}s @ {V_PULSE}V)')
    plt.axvspan(T_PULSE, t[-1], color='gray', alpha=0.1,
                label=f'Relajación ({V_READ}V)')

    plt.title('Dinámica de Conductancia: Fase de Facilitación y Relajación',
              fontsize=14)
    plt.xlabel('Tiempo (s)', fontsize=12)
    plt.ylabel('Conductancia (mS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    ymax = np.max(G) if G.size > 0 else 1.0
    mid_pulse = T_PULSE / 2
    mid_relax = T_PULSE + (t[-1] - T_PULSE) / 2

    plt.annotate('Potenciación\n(Facilitation)',
                 xy=(mid_pulse, ymax * 0.5),
                 ha='center', fontweight='bold', color='orange')
    plt.annotate('Decaimiento\n(Relaxation)',
                 xy=(mid_relax, ymax * 0.2),
                 ha='center', fontweight='bold', color='blue')

    plt.tight_layout()
    plt.show()