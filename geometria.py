import numpy as np
import matplotlib.pyplot as plt

# CONSTANTES DE GEOMETRÍA POR DEFECTO
NUM_WIRES = 2000
AREA = 1000.0       
LENGTH = 70.0       




def generate_and_find_junctions(num_wires, wire_length, area_size):
    '''
    La función se encarga de simular la disposición de nanohilos en un área cuadrada y encontrar todos los puntos donde estos nanohilos se cruzan.
    Genera "num_wires" nanohilos de longitud "wire_length" dentro de un cuadrado de lado "area_size" y despues detecta todas las intersecciones entre ellos.

    devuelve tres estructuras principales:

    *wires*:
        Es una lista de diccionarios, donde cada diccionario representa un nanohilo.
        Cada nanohilo tiene:

        'id': Un identificador único para el nanohilo.
        'p1': Un array NumPy que representa las coordenadas (x, y) del primer extremo del nanohilo.
        'p2': Un array NumPy que representa las coordenadas (x, y) del segundo extremo del nanohilo.


    *junctions*:
        Es una lista de diccionarios, donde cada diccionario representa un punto de cruce o unión entre dos nanohilos.
        Cada unión contiene:

        'id': Un identificador único para la unión.
        'pos': Un array NumPy con las coordenadas (x, y) exactas del punto de intersección.
        'wires': Una tupla con los IDs de los dos nanohilos que se cruzan en esa unión.


    *wire_to_junctions*:
        Es un diccionario que mapea el ID de cada nanohilo a una lista de todas las uniones en las que participa ese nanohilo.
        Esto es útil para navegar por las uniones a lo largo de un nanohilo específico.
    '''


    xc = np.random.uniform(0, area_size, num_wires)
    yc = np.random.uniform(0, area_size, num_wires)
    theta = np.random.uniform(0, np.pi, num_wires)

    x_off = (wire_length / 2) * np.cos(theta)
    y_off = (wire_length / 2) * np.sin(theta)

    wires = []
    for i in range(num_wires):
        wires.append({'id': i, 'p1': np.array((xc[i] - x_off[i], yc[i] - y_off[i])), 'p2': np.array((xc[i] + x_off[i], yc[i] + y_off[i]))})

    junctions, wire_to_junctions = [], {i: [] for i in range(num_wires)}
    junction_id_counter = 0

    # Optimización simple: usar bounding boxes si fuera necesario, aquí fuerza bruta
    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            w1, w2 = wires[i], wires[j]
            d1 = w1['p2'] - w1['p1']; d2 = w2['p2'] - w2['p1']
            denom = d1[0]*d2[1] - d1[1]*d2[0]
            if denom != 0:
                t = ((w2['p1'][0] - w1['p1'][0])*d2[1] - (w2['p1'][1] - w1['p1'][1])*d2[0]) / denom
                u = ((w2['p1'][0] - w1['p1'][0])*d1[1] - (w2['p1'][1] - w1['p1'][1])*d1[0]) / denom
                if 0 <= t <= 1 and 0 <= u <= 1:
                    ix, iy = w1['p1'] + t * d1
                    j_data = {'id': junction_id_counter, 'pos': np.array([ix, iy]), 'wires': (i, j)}
                    junctions.append(j_data)
                    wire_to_junctions[i].append(j_data); wire_to_junctions[j].append(j_data)
                    junction_id_counter += 1
    return wires, junctions, wire_to_junctions




def plot_physical_network(wires, junctions, area_size=AREA):
    """ Dibuja la red física de hilos y sus junturas en el plano 2D """
    plt.figure(figsize=(8, 8))
    for w in wires:
        plt.plot([w['p1'][0], w['p2'][0]], [w['p1'][1], w['p2'][1]], color='gray', alpha=0.6, lw=1)
    
    if len(junctions) > 0:
        j_positions = np.array([j['pos'] for j in junctions])
        plt.scatter(j_positions[:, 0], j_positions[:, 1], color='red', s=5, zorder=3, label='Junturas')
        
    plt.xlim(0, area_size)
    plt.ylim(0, area_size)
    plt.title(f"Red Física Estocástica ({len(wires)} hilos, {len(junctions)} junturas)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()