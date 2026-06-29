# neuromorphicNWLamas

# 🧠 Simulador de Redes de Nanohilos Neuromórficas (Nanowire Networks - NWN)

Este repositorio contiene una librería modular en Python para la generación estocástica, modelado topológico y simulación eléctrica/dinámica de redes de nanohilos de plata (Ag). El sistema simula el comportamiento de "memristores" stocásticos autoensamblados en los puntos de intersección de los filamentos, emulando la plasticidad sináptica y los procesos de memoria del cerebro humano.

---

## 🗺️ Arquitectura de la Librería

El código se encuentra desacoplado en 4 módulos independientes y altamente optimizados:

1. **`geometria.py`**: Controla el modelado físico y espacial de la deposición de nanohilos en el plano 2D, calculando las coordenadas exactas de las junturas.
2. **`grafo.py`**: Transforma la red física en una abstracción matemática basada en grafos (`networkx`) mediante un algoritmo exacto de duplicación de nodos.
3. **`fisica.py`**: Construye las ecuaciones circuitales basadas en las Leyes de Kirchhoff y actualiza estocásticamente las conductancias según los voltajes locales.
4. **`simulador.py`**: El motor temporal maestro que integra los módulos para correr simulaciones dinámicas y ciclos de histéresis completos.

---

## 🛠️ Documentación Detallada de Funciones

### 📦 1. Módulo: `geometria.py`
Se encarga de la generación espacial estocástica de los nanocables sobre el sustrato y de la detección analítica de colisiones geométricas.

#### `generate_and_find_junctions(num_wires, wire_length, area_size)`
Genera una distribución uniforme de hilos con centros aleatorios $(x_c, y_c)$ y ángulos de orientación $\theta \in [0, \pi)$. Luego, aplica un algoritmo de fuerza bruta analítico para encontrar los cruces de segmentos.
* **Entradas:**
    * `num_wires` *(int)*: Cantidad total de nanohilos a depositar (Densidad de la red).
    * `wire_length` *(float)*: Longitud física de cada nanocable (en $\mu m$ o unidades relativas).
    * `area_size` *(float)*: Tamaño del lado del sustrato cuadrado.
* **Salidas:**
    * `wires` *(list)*: Diccionarios con los puntos extremos `p1` y `p2` de cada hilo.
    * `junctions` *(list)*: Diccionarios con el `id` de la juntura, posición exacta `pos` $[x, y]$ e índices de los dos hilos involucrados.
    * `wire_to_junctions` *(dict)*: Mapa de adyacencia que asocia cada hilo con todas las junturas que posee.

#### `plot_physical_network(wires, junctions, area_size)`
Genera una representación visual en 2D de la red física utilizando `matplotlib`.
* **Entradas:** Lista de hilos, lista de junturas y dimensiones del área. Muestra los nanohilos en líneas grises y resalta los puntos memristivos (junturas) en rojo.

---

### 📦 2. Módulo: `grafo.py`
Mapea la estructura geométrica a un objeto de la librería `networkx`. Implementa una **topología por duplicación de nodos** para modelar correctamente las junturas memristivas como aristas (edges).

#### `build_graph2(wires, junctions, wire_to_junctions, G_OFF_default)`
Construye el grafo de admitancia. Por cada juntura física, genera dos nodos en el grafo (uno para cada cara del hilo que cruza). La arista que une estos nodos duplicados representa el comportamiento memristivo de la juntura. Los segmentos internos del nanohilo que conectan una juntura con otra se representan como aristas resistivas lineales ordinarias.
* **Entradas:** Salidas de la etapa geométrica y el valor inicial de la conductancia en estado apagado (`G_OFF`).
* **Salidas:** `G` *(nx.Graph)*: Grafo topológico estructural.

#### `find_electrode_nodes2(G, area_size, threshold)`
Identifica qué nodos del grafo están lo suficientemente cerca de las fronteras izquierda ($x=0$) y derecha ($x=\text{area\_size}$) para actuar como terminales eléctricos (Ánodo/Cátodo).
* **Entradas:** Grafo, tamaño del área y el umbral físico de proximidad (`threshold`).
* **Salidas:** `input_nodes` *(list)*, `output_nodes` *(list)*: Nodos frontera.

#### `check_percolation(G, input_nodes, output_nodes)`
Verifica si existe conectividad topológica estructural entre los electrodos de entrada y salida utilizando algoritmos de búsqueda en grafos.
* **Salidas:** `True` si hay un camino continuo de transporte, `False` en caso contrario.

#### `get_shortest_path_length(G, input_nodes, output_nodes)`
Calcula la longitud topológica (número de saltos/nodos) del camino más corto que conecta los extremos a través de la red.

---

### 📦 3. Módulo: `fisica.py`
Resuelve la física de transporte eléctrico aplicando conservación de la carga en los nodos y leyes de conmutación probabilísticas.

#### `build_admittance_matrix2(G, V_INPUT, V_GROUND, input_nodes, output_nodes, r_wire_per_length, g_off_default)`
Ensambla la matriz de admitancia del circuito disperso ($Y$) y el vector de corrientes externas ($I$) para plantear el sistema de ecuaciones lineales de Kirchhoff ($Y \cdot V = I$).
* Calcula la resistencia inherente de los nanocables de plata en base a su resistividad volumétrica ($\rho_{\text{Ag}}$) y sección transversal.
* Aplica condiciones de contorno de Dirichlet fijando los voltajes en los nodos de los electrodos e inyecta una conductancia de fuga mínima (`G_LEAK`) para evitar singularidades matemáticas.
* **Salidas:** Matriz CSR dispersa `Y`, vector `I_vec` y el diccionario de indexación de nodos `node_to_index`.

#### `update_stochastic_conductance2(G, V_solved, node_to_index, G_ON_val, G_OFF_val, V_THRESHOLD_val, P0_SET_val, ALPHA_SET_val, P_DECAY_val)`
Aplica el modelo dinámico de conmutación. Evalúa el módulo del voltaje local $|V_{\text{mem}}|$ en cada juntura:
1.  **Proceso SET (OFF $\rightarrow$ ON):** Si $|V_{\text{mem}}| > V_{\text{th}}$, la probabilidad de tunelamiento o formación de filamentos aumenta exponencialmente: $P_{\text{set}} = P_0 \cdot \exp(\alpha \cdot (|V_{\text{mem}}| - V_{\text{th}}))$.
2.  **Proceso RESET (ON $\rightarrow$ OFF):** Si el voltaje decae, el filamento colapsa de forma estocástica guiado por un factor de decaimiento térmico (`P_DECAY`).
* **Salidas:** Retorna el objeto `G` con los estados de conductancia actualizados.

#### `calculate_input_current(G, V_solved, node_to_index, input_nodes, r_wire_per_length)`
Calcula la corriente neta total que fluye a través del electrodo de entrada sumando las contribuciones individuales de corriente de todas las ramas frontera conectadas.

---

### 📦 4. Módulo: `simulador.py`
Funciona como el orquestador del tiempo discreto ($dt$) ejecutando los bucles de simulación física.

#### `run_simulation_dynamic2(num_wires_to_simulate)`
Simula la respuesta temporal de la red ante un escalón de voltaje continuo constante (`V_INPUT`). Mide la evolución de la conductancia equivalente total del sistema y cuenta cuántos memristores internos se activaron a lo largo del tiempo.
* **Salidas:** Historiales de tiempo, conductancia total del sistema y cantidad de junturas activas en estado `ON`.

#### `run_simulation_dynamic_histeresis(num_wires_to_simulate, cantidad_ciclos)`
Somete a la red a una señal de excitación alterna sinusoidal $V(t) = V_0 \cdot \in(2\pi f t)$. Permite extraer las curvas $I-V$ típicas de los sistemas con memoria.
* **Salidas:** Historiales temporales de voltaje de entrada, corriente total inyectada y canales memristivos activos (ideal para graficar la ventana de histéresis "pinched hysteresis loop").

---

## 🚀 Uso en Google Colab

Para clonar la librería en tu entorno de Colab y correr un experimento, ejecutá:

```python
import sys
# Clonar el repositorio público
!git clone [https://github.com/tu_usuario/neuromorphicNWLamas.git](https://github.com/tu_usuario/neuromorphicNWLamas.git)
sys.path.append('/content/neuromorphicNWLamas')

# Ejecutar un experimento de histéresis
import simulador as sim
import matplotlib.pyplot as plt

t, v, i, act = sim.run_simulation_dynamic_histeresis(num_wires_to_simulate=1500, cantidad_ciclos=2)

# Graficar la curva de histéresis característica
plt.figure(figsize=(6, 5))
plt.plot(v, i, color='purple')
plt.title("Lazo de Histéresis de la Red")
plt.xlabel("Voltaje (V)")
plt.ylabel("Corriente (A)")
plt.grid(True)
plt.show()
