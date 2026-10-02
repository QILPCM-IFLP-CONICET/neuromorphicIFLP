# Modelos de evolución de memristores

## Resumen

Al final de cada paso temporal del motor (`run_simulation_dynamic_pulse`),
el estado ON/OFF de cada memristor se actualiza invocando una función
*evolver*. Qué función se usa se selecciona con la clave `evolver_model`
del `.ini` —o su equivalente `EVOLVER` en los overrides— y la resolución
se hace contra un registro global:

~~~python
update_conductance = fis.EVOLVE_MODELS[p["EVOLVER"]]
~~~

El paquete expone un mecanismo de registro para agregar modelos propios
sin modificar el código fuente.

## Selección del modelo

### Desde un `.ini`

~~~ini
[Memristor]
g_on = 1e-3
g_off = 1e-9
v_threshold = 0.01
evolver_model = stochastic1
~~~

### Desde `parms=`

~~~python
sim = setup_simulation(parms={"EVOLVER": "stochastic1"})
~~~

El valor debe coincidir con una clave ya registrada en
`neuromorphic.fisica.evolvers.EVOLVE_MODELS`. Si no existe, el motor
falla con `KeyError` en el primer paso temporal: la búsqueda es *lazy*,
no hay validación temprana en `setup_simulation`.

## Contrato de un evolver

~~~python
def mi_evolver(
    simulation: dict[str, Any],
    V_solved: np.ndarray,
) -> "networkx.Graph":
    ...
~~~

Recibe el diccionario completo de simulación y el vector de voltajes
nodales recién resuelto (`Y · V = I`). Debe **mutar in-place** dos
estructuras de forma coherente:

| Estructura | Descripción |
|---|---|
| `simulation["circuit"]["memristor_g"]` | Array `float64` con la conductancia actual de cada memristor. Lo consume `build_admittance_matrix`. |
| `simulation["graph"].edges[u, v]["conductance"]` | Valor espejo en las aristas del grafo. Lo usan diagnósticos y visualización. |

También están disponibles:

- `simulation["parameters"]` — parámetros crudos y derivados.
- `simulation["circuit"]["mem_edge_keys"]` — lista de tuplas `(u, v)` en
  el mismo orden que `memristor_g`.
- `simulation["circuit"]["mem_u_idx"]`, `["mem_v_idx"]` — índices en
  `V_solved` de los extremos de cada memristor.
- `V_solved` — voltajes nodales del paso actual.

El valor de retorno se ignora; se recomienda devolver el grafo por
convención.

> **RNG**: si el evolver consume `numpy.random`, altera el estado global
> del generador. Dos evolvers distintos con la misma semilla no producirán
> necesariamente las mismas conmutaciones; la distribución estadística del
> proceso sí es la misma. Si necesitás reproducibilidad estricta, pasá tu
> propio `np.random.Generator` a través de `simulation["parameters"]`.

## Registrar un modelo propio

~~~python
import numpy as np
from neuromorphic.fisica.evolvers import register_memristor_evol_model


@register_memristor_evol_model("siempre_on")
def siempre_on(simulation, V_solved):
    """Ejemplo trivial: fuerza todos los memristores a ON."""
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]

    # 1) Array consumido por build_admittance_matrix
    circuit["memristor_g"][:] = p["G_ON"]

    # 2) Espejo en el grafo
    for u, v in circuit["mem_edge_keys"]:
        G.edges[u, v]["conductance"] = p["G_ON"]

    return G
~~~

Una vez importado el módulo que contiene el decorador, el modelo queda
disponible para cualquier simulación del proceso:

~~~python
from neuromorphic import setup_simulation, run_simulation_dynamic_pulse

sim = setup_simulation(parms={"EVOLVER": "siempre_on"})
t, G_total, activos = run_simulation_dynamic_pulse(sim)
~~~

El registro es **global al proceso**: importá el módulo del evolver una
sola vez (por ejemplo en tu `__init__.py` o al inicio del notebook) antes
de llamar a `setup_simulation`.

## Modelos incluidos

| Nombre | Descripción |
|---|---|
| `stochastic1` | SET/RESET estocástico con facilitación y relajación volátil (Lamas et al., 2026). Modelo por defecto. |

## Ver también

- [`arquitectura.md`](arquitectura.md) — flujo de datos entre módulos.
- Código fuente: `src/neuromorphic/fisica/evolvers/base.py`.