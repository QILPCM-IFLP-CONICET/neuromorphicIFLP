# neuromorphicNWLamas

Simulador estocástico de redes de nanohilos neuromórficas (Nanowire
Networks, NWN) en Python. Genera redes autoensambladas de nanohilos de
plata, las modela como grafos con junturas memristivas y resuelve su
dinámica eléctrica mediante las leyes de Kirchhoff, incluyendo efectos
de facilitación y relajación volátil inspirados en plasticidad sináptica.

---

## Instalación

Requiere Python 3.10 o superior.

```bash
git clone https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas.git
cd neuromorphicNWLamas
pip install -e .
```

Para desarrollo (tests, lint, type-check):

```bash
pip install -e ".[dev]"
```

---

## Uso rápido

```python
from neuromorphic import setup_simulation, run_simulation_dynamic_pulse
from neuromorphic.visualizacion import plot_simulation_results

# Construye la simulación con los parámetros por defecto del paquete.
sim = setup_simulation()

# Corre el experimento de pulso y relajación (Fig. 2b del paper).
t, G_total, active = run_simulation_dynamic_pulse(sim)

# Grafica los resultados.
plot_simulation_results(sim, t, G_total)
```

### Overrides sin tocar archivos

Todas las variantes se controlan con un diccionario `parms` que se
aplica sobre los valores por defecto:

```python
sim = setup_simulation(parms={
    "NUM_WIRES": 1300,
    "T_PULSE": 5.0,
    "T_RELAX": 20.0,
    "TIME_STEP_DT": 1e-3,
})
```

Las claves desconocidas en `parms` emiten `UnknownParameterWarning` y se
ignoran. Los valores derivados (`PROXIMITY_THRESHOLD`,
`R_WIRE_PER_LENGTH`, `TOTAL_TIME`) se recalculan automáticamente después
del merge.

### Configuración con archivo propio

```python
sim = setup_simulation(filepath="mi_config.ini")
```

El archivo `defaults.ini` empaquetado con la librería sirve de plantilla.

---

## Modelo físico

Cada nanohilo se representa como un segmento de longitud `L` con
resistencia distribuida. En cada cruce entre dos hilos se coloca un
**memristor** cuya conductancia conmuta estocásticamente entre dos
estados:

| Estado | Conductancia | Conmutación     |
|--------|--------------|-----------------|
| OFF    | `G_OFF`      | SET por voltaje |
| ON     | `G_ON`       | RESET por corriente |

La actualización sigue las probabilidades reportadas en Lamas et al.
(2026):

- **SET**: P_set = Δt · P₀ · max(0, 1 − exp[−α(V_mem − V_th)])
- **RESET**: P_reset = min(1, Δt · P_decay · I_mem²)

En cada paso temporal se resuelve el sistema Y · V = I sobre la matriz
de admitancia dispersa del grafo, aplicando condiciones de contorno de
Dirichlet en los electrodos.

La red presenta un **umbral de percolación** en torno a N ≈ 1200 hilos
para L = 70 µm y A = 1 mm². Por debajo de ese valor no existe camino
topológico entre electrodos y la simulación se rechaza con
`RuntimeError`.

---

## Estructura del paquete

```
src/neuromorphic/
├── defaults.ini       # Parámetros por defecto (empaquetados)
├── load_config.py     # Carga y validación de parámetros
├── geometria.py       # Generación espacial de hilos y junturas
├── grafo.py           # Topología (networkx) y electrodos
├── fisica.py          # Matriz de admitancia y dinámica estocástica
├── simulador.py       # Motor temporal (pulso, relajación)
├── visualizacion.py   # Gráficas científicas
└── simulation.py      # Entry point: setup_simulation()
```

La API pública se expone desde el paquete:

```python
from neuromorphic import (
    setup_simulation,
    run_simulation_dynamic_pulse,
    plot_simulation_results,
    update_stochastic_conductance,
)
```

---

## Tests

```bash
pytest
```

La suite corre en aproximadamente 10 segundos e incluye tests unitarios
por módulo e integración end-to-end con redes por encima y por debajo
del umbral de percolación.

---

## Referencia

Este código implementa el modelo descrito en:

> Lamas et al. (2026). *Stochastic Modeling of Silver Nanowire Networks
for Neuromorphic Computing*. 55º Jornadas Argentinas de
> Informática (JAIiO).
> https://55jaiio.sadio.org.ar/wp-content/uploads/2026/07/151.pdf



---

## Licencia

MIT. Ver [LICENSE](LICENSE).

[![Tests](https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas/actions/workflows/tests.yml/badge.svg)](...)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](...)
