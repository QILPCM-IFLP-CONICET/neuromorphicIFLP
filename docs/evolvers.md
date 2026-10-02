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

El valor debe coincidir con un modelo registrado en
`neuromorphic.fisica.EVOLVER_SPECS`. Si no existe, `setup_simulation`
falla con `KeyError` y la lista de modelos disponibles.

## Componentes de un modelo

Un modelo se registra con

~~~python
@register_memristor_evol_model(name, init=None, parameters=None)
def update(simulation, V_solved): ...
~~~

y queda descrito por un `EvolverSpec(update, init, parameters)`:

| Componente | Rol |
|---|---|
| `update(simulation, V_solved)` | Avanza un paso temporal. Obligatorio. |
| `init(simulation)` | Fija el estado inicial y precalcula lo que el modelo necesite. Opcional; por defecto, todos los memristores en `G_OFF`. |
| `parameters` | Dict `{CLAVE: valor_por_defecto}` con los parámetros propios del modelo. Opcional. |

### Ciclo de vida

`initialize_evolver(simulation)` prepara el modelo indicado en
`parameters["EVOLVER"]`:

1. completa en `simulation["parameters"]` las claves de `parameters` que
   falten, sin pisar valores ya definidos por el `.ini` o por `parms=`;
2. recrea `simulation["evolver_state"]` como un dict vacío;
3. llama a `init(simulation)`.

La llaman `setup_simulation`, al final, y `run_simulation_dynamic_pulse`,
al comienzo de cada corrida. Por eso correr dos veces la misma
simulación parte siempre del mismo estado inicial.

### Parámetros propios

Las claves declaradas en `parameters` se aceptan en `parms=` sin
advertencias y conviven con el resto en `simulation["parameters"]`.
Pueden ser escalares o arrays con un valor por memristor (en el orden de
`memristor_g`). También pueden definirse en el `.ini`, en una sección
libre `[Evolver]`: cada clave se pasa a mayúsculas y su valor se
interpreta como literal de Python.

~~~ini
[Evolver]
nu_up = 1e3
max_jumps = 2
~~~

## Contrato de `update`

~~~python
def mi_evolver(
    simulation: dict[str, Any],
    V_solved: np.ndarray,
) -> "networkx.Graph":
    ...
~~~

Recibe el diccionario completo de simulación y el vector de voltajes
nodales recién resuelto (`Y · V = I`). Debe mantener coherentes tres
estructuras (`init` también debe dejarlas coherentes):

| Estructura | Descripción |
|---|---|
| `simulation["circuit"]["memristor_g"]` | Array `float64` con la conductancia actual de cada memristor. Lo consume `build_admittance_matrix`. Se modifica in-place. |
| `simulation["graph"].edges[u, v]["conductance"]` | Valor espejo en las aristas del grafo. Lo usan el cálculo de corrientes, diagnósticos y visualización. |
| `simulation["circuit"]["memristor_active"]` | Máscara booleana de memristores "activos". El motor reporta su suma en cada paso. Para modelos binarios es `memristor_g == G_ON`; un modelo de conductancia continua define su propio criterio. |

El estado interno del modelo (variables por juntura, tablas
precalculadas, generadores aleatorios, diagnósticos) va en
`simulation["evolver_state"]`.

También están disponibles:

- `simulation["parameters"]` — parámetros crudos, derivados y propios del modelo.
- `simulation["circuit"]["mem_edge_keys"]` — lista de tuplas `(u, v)` en
  el mismo orden que `memristor_g`.
- `simulation["circuit"]["mem_u_idx"]`, `["mem_v_idx"]` — índices en
  `V_solved` de los extremos de cada memristor.
- `V_solved` — voltajes nodales del paso actual.

El valor de retorno se ignora; se recomienda devolver el grafo por
convención.

> **RNG**: `setup_simulation` siembra el generador global con
> `np.random.seed(RNG_SEED)`, y el evolver incluido (`stochastic1`)
> consume `numpy.random` directamente. Con la misma semilla y el mismo
> evolver la corrida es reproducible. Dos evolvers distintos con la misma
> semilla no producirán necesariamente las mismas conmutaciones, porque
> consumen el generador en distinto orden. Por ahora el paquete no admite
> inyectar un `np.random.Generator` global; si tu modelo lo necesita,
> crealo en `init` (por ejemplo con `np.random.default_rng(p["RNG_SEED"])`)
> y guardalo en `simulation["evolver_state"]`.

## Registrar un modelo propio

El ejemplo siguiente usa los tres componentes: un parámetro propio
(`P_FLIP`), un `init` que crea un generador aleatorio en
`evolver_state`, y un `update` que mantiene coherentes array, grafo y
máscara.

~~~python
import numpy as np
from neuromorphic.fisica import register_memristor_evol_model


def init_flip(simulation):
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]
    circuit["memristor_g"][:] = p["G_OFF"]
    for u, v in circuit["mem_edge_keys"]:
        G.edges[u, v]["conductance"] = p["G_OFF"]
    circuit["memristor_active"] = np.zeros(len(circuit["memristor_g"]), dtype=bool)
    simulation["evolver_state"]["rng"] = np.random.default_rng(p["RNG_SEED"])


@register_memristor_evol_model("flip", init=init_flip, parameters={"P_FLIP": 0.01})
def flip(simulation, V_solved):
    """Ejemplo trivial: cada memristor invierte su estado con prob. P_FLIP."""
    p = simulation["parameters"]
    circuit = simulation["circuit"]
    G = simulation["graph"]
    rng = simulation["evolver_state"]["rng"]

    mem_g = circuit["memristor_g"]
    flip = rng.random(mem_g.size) < p["P_FLIP"]
    mem_g[flip] = np.where(mem_g[flip] == p["G_ON"], p["G_OFF"], p["G_ON"])
    for j in np.flatnonzero(flip):
        u, v = circuit["mem_edge_keys"][j]
        G.edges[u, v]["conductance"] = mem_g[j]
    circuit["memristor_active"] = mem_g == p["G_ON"]
    return G
~~~

Para el caso común de fijar todos los memristores en un mismo valor
existe el helper `set_all_memristors(simulation, g_value)`, que actualiza
array, grafo y máscara.

Una vez importado el módulo que contiene el decorador, el modelo queda
disponible para cualquier simulación del proceso:

~~~python
from neuromorphic import setup_simulation, run_simulation_dynamic_pulse

sim = setup_simulation(parms={"EVOLVER": "flip", "P_FLIP": 0.05})
t, G_total, activos = run_simulation_dynamic_pulse(sim)
~~~

El registro es **global al proceso**: importá el módulo del evolver una
sola vez (por ejemplo en tu `__init__.py` o al inicio del notebook) antes
de llamar a `setup_simulation`, para que sus parámetros se reconozcan
como declarados. Registrar un nombre que ya
existe reemplaza la función anterior sin emitir advertencia.

## Modelos incluidos

| Nombre | Descripción |
|---|---|
| `stochastic1` | SET por sobretensión con P = P0·exp[α(V−V_th)] y RESET con P = P_decay·exp(−V/V_th). Las probabilidades son por paso y no escalan con `TIME_STEP_DT`. Modelo por defecto. |
| `stochastic2` | Modelo de Lamas et al. (2026): P_set = Δt·P0·max(0, 1 − exp[−α(V−V_th)]) y P_reset = min(1, Δt·P_decay·I²), con I = G_ON·V. `P0_SET` está en s⁻¹ y `P_DECAY` en A⁻²·s⁻¹. |

> Los valores de `defaults.ini` están calibrados para `stochastic1`. Con
> `stochastic2` las mismas cifras tienen otras unidades: por ejemplo,
> con `G_ON = 1e-3` S y caídas de ~1 V, `P_DECAY = 0.8` da una
> probabilidad de RESET del orden de 10⁻⁹ por paso.

## Ver también

- [`arquitectura.md`](arquitectura.md) — flujo de datos entre módulos.
- Código fuente: `src/neuromorphic/fisica/evolvers/base.py`.