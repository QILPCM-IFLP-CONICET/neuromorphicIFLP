import numpy as np
import matplotlib.pyplot as plt

# CONSTANTES DE GEOMETRÍA POR DEFECTO
NUM_WIRES = 2000
AREA = 1000.0       
LENGTH = 70.0       

def generate_and_find_junctions(num_wires=NUM_WIRES, wire_length=LENGTH, area_size=AREA):
    """
    Genera la disposición de nanohilos y calcula analíticamente todas las intersecciones.
    """
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