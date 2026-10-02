# Arquitectura

## Flujo de datos

~~~text
                 defaults.ini  (parámetros empaquetados)
                       │
                       ▼
                 load_config.py  ──►  simulation["parameters"]
                       │
        ┌──────────────┼──────────────┬──────────────┐
        ▼              ▼              ▼              ▼
   geometria.py    grafo.py       fisica/      visualizacion.py
   (hilos y        (topología,    ├── admitancia.py   (gráficas)
    junturas)       electrodos)   ├── corrientes.py
                                   └── evolvers/
                                       └── EVOLVE_MODELS
                                           ◄── register_memristor_evol_model
        │              │              │              │
        └──────────────┴──────────────┘              │
                       ▼                             │
                 dinamica.py  ◄──────────────────────┘
                 (motor temporal)
                       │
                       ▼
            setup_simulation() / run_*
            (interfaz pública del paquete)
~~~

## Enfoque de parámetros

Todos los parámetros se cargan **una sola vez** en un diccionario
`simulation` con la estructura:

~~~python
simulation = {
    "parameters": {...},   # crudos + derivados
    "junctions": {...},    # geometría de la red
    "graph": <networkx.Graph>,
    "terminals": {...},    # electrodos de entrada/salida
    "circuit": {...},      # matriz Y, vector I, mapeo de nodos
}
~~~

Cada función recibe `simulation` y lee lo que necesita de ahí. No hay
variables globales ni parámetros leídos implícitamente del directorio
de trabajo. Los overrides se hacen vía `parms=` en `setup_simulation` o
`load_parameters`.

## Uso en Google Colab

~~~python
!pip install git+https://github.com/QILPCM-IFLP-CONICET/neuromorphicNWLamas.git

import configparser, importlib, sys
import ipywidgets as widgets
from IPython.display import display, clear_output

import neuromorphic
from neuromorphic import setup_simulation, run_simulation_dynamic_pulse
from neuromorphic.visualizacion import plot_simulation_results

# Panel interactivo: los sliders sobrescriben los parámetros en cada corrida.
w_num_wires = widgets.IntSlider(value=1300, min=800, max=3000, step=100, description='Nº hilos:')
w_area      = widgets.FloatSlider(value=1000.0, min=500.0, max=2000.0, step=100.0, description='Área (µm):')
w_v_input   = widgets.FloatSlider(value=3.6, min=1.0, max=10.0, step=0.1, description='V pulso (V):')
w_diametro  = widgets.FloatSlider(value=115.0, min=50.0, max=200.0, step=5.0, description='Ø NW (nm):')

btn = widgets.Button(description="Correr simulación", button_style='success')
out = widgets.Output()

def on_run(_):
    with out:
        clear_output(wait=True)
        sim = setup_simulation(parms={
            "NUM_WIRES": w_num_wires.value,
            "AREA": w_area.value,
            "V_INPUT": w_v_input.value,
            "DIAMETRO_NM": w_diametro.value,
        })
        t, g, _ = run_simulation_dynamic_pulse(sim)
        plot_simulation_results(sim, t, g)

btn.on_click(on_run)
display(widgets.HBox([widgets.VBox([w_num_wires, w_area]),
                      widgets.VBox([w_v_input, w_diametro])]))
display(btn, out)
~~~

Con el paquete instalable, **no hace falta clonar ni manipular
`sys.path`**: `pip install` trae el paquete, el `defaults.ini` y los
tests. Los overrides van todos por `parms=`.

## Registro de evolvers

`fisica/evolvers/base.py` mantiene un diccionario global
`EVOLVE_MODELS: dict[str, Callable]`. Cada modelo se registra al momento
de importar su módulo mediante el decorador
`register_memristor_evol_model("nombre")`:

~~~python
EVOLVE_MODELS = {}

def register_memristor_evol_model(name: str):
    def _register(fn):
        EVOLVE_MODELS[name] = fn
        return fn
    return _register
~~~

El motor temporal resuelve `p["EVOLVER"]` contra ese diccionario en cada
corrida. La clave se lee del `.ini` como `evolver_model` en la sección
`[Memristor]`, y puede sobrescribirse con `parms={"EVOLVER": "..."}`.

Ver [`evolvers.md`](evolvers.md) para el contrato completo, la lista de
modelos incluidos y ejemplos de implementaciones propias.