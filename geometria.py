# geometria.py
import numpy as np
from load_config import cargar_parametros

# ==============================================================================
# CARGA DE PARAMETROS DESDE CONFIG.INI (Enfoque A)
# ==============================================================================
p = cargar_parametros()


# ==============================================================================
# FUNCIONES DE GEOMETRÍA Y RED
# ==============================================================================
def generate_and_find_junctions():
    '''
    La función se encarga de simular la disposición de nanohilos en un área cuadrada y encontrar todos los puntos donde estos nanohilos se cruzan.
    Usa los parámetros centralizados en config.ini a través del diccionario 'p'.

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
    # Extraemos las variables del diccionario centralizado
    num_wires = p['NUM_WIRES']
    wire_length = p['LENGTH']
    area_size = p['AREA']

    # Generación de posiciones aleatorias basándonos en el substrato
    xc = np.random.uniform(0, area_size, num_wires)
    yc = np.random.uniform(0, area_size, num_wires)
    theta = np.random.uniform(0, np.pi, num_wires)

    x_off = (wire_length / 2) * np.cos(theta)
    y_off = (wire_length / 2) * np.sin(theta)

    wires = []
    for i in range(num_wires):
        wires.append({
            'id': i, 
            'p1': np.array((xc[i] - x_off[i], yc[i] - y_off[i])), 
            'p2': np.array((xc[i] + x_off[i], yc[i] + y_off[i]))
        })

    junctions, wire_to_junctions = [], {i: [] for i in range(num_wires)}
    junction_id_counter = 0

    # Detección de intersecciones por fuerza bruta
    for i in range(num_wires):
        for j in range(i + 1, num_wires):
            w1, w2 = wires[i], wires[j]
            d1 = w1['p2'] - w1['p1']
            d2 = w2['p2'] - w2['p1']
            denom = d1[0]*d2[1] - d1[1]*d2[0]
            if denom != 0:
                t = ((w2['p1'][0] - w1['p1'][0])*d2[1] - (w2['p1'][1] - w1['p1'][1])*d2[0]) / denom
                u = ((w2['p1'][0] - w1['p1'][0])*d1[1] - (w2['p1'][1] - w1['p1'][1])*d1[0]) / denom
                if 0 <= t <= 1 and 0 <= u <= 1:
                    ix, iy = w1['p1'] + t * d1
                    j_data = {'id': junction_id_counter, 'pos': np.array([ix, iy]), 'wires': (i, j)}
                    junctions.append(j_data)
                    wire_to_junctions[i].append(j_data)
                    wire_to_junctions[j].append(j_data)
                    junction_id_counter += 1
                    
    return wires, junctions, wire_to_junctions