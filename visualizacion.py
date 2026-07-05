


import numpy as np
import matplotlib.pyplot as plt



# ==============================================================================
# CARGA DE PARAMETROS DESDE CONFIG.INI (Enfoque A)
# ==============================================================================
p = cargar_parametros()



def plot_simulation_results(history_time, history_G_total):
    '''
    Genera la grafica de la evolucion temporal de la conductancia de la red,
    reproduciendo el comportamiento dinamico de facilitacion y relajacion volatil
    (inspirado en la Fig. 2b del paper de referencia de Milano ... agregar cita).

    La funcion divide visualmente el grafico en dos regiones mediante sombreados:
    1. Fase de Pulso (Potenciación): Muestra el aumento de conductancia ante un voltaje alto.
    2. Fase de Relajación (Decaimiento): Muestra la disolución estocástica de los 
       filamentos y la consecuente caída de conductancia bajo un voltaje de lectura mínimo.

    Parámetros:
    -----------
    history_time : list o numpy.ndarray
        Vector o lista con los pasos de tiempo (en segundos) registrados en la simulación.
    history_G_total : list o numpy.ndarray
        Vector o lista con los valores de la conductancia total equivalente de la red 
        (en miliSiemens, mS) calculados en cada paso temporal.

    Devuelve:
    ---------
    None
        La función no retorna ningún valor. Renderiza y muestra el gráfico directamente 
        en pantalla utilizando matplotlib.
    '''



    # Convertir a arrays de numpy para facilitar el manejo
    t = np.array(history_time)
    G = np.array(history_G_total)

    plt.figure(figsize=(10, 6))

    # Dibujamos la curva de conductancia
    plt.plot(t, G, color='#2c3e50', linewidth=2, label='Conductancia de la Red')

    # Sombreado para identificar las fases (como en el paper)
    plt.axvspan(0, 10, color='yellow', alpha=0.2, label='Pulso (10s @ 4V)')
    plt.axvspan(10, t[-1], color='gray', alpha=0.1, label='Relajación (0.05V)')

    # Configuración de ejes y estilo
    plt.title('Dinámica de Conductancia: Fase de Facilitación y Relajación', fontsize=14)
    plt.xlabel('Tiempo (s)', fontsize=12)
    plt.ylabel('Conductancia (mS)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Anotaciones para clarificar la Fig 2-b
    plt.annotate('Potenciación\n(Facilitation)', xy=(5, np.max(G)*0.5),
                 ha='center', fontweight='bold', color='orange')
    plt.annotate('Decaimiento\n(Relaxation)', xy=(100, np.max(G)*0.2),
                 ha='center', fontweight='bold', color='blue')

    plt.tight_layout()
    plt.show()

