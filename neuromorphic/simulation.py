# simulation.py
from typing import Optional

from .load_config import cargar_parametros
from .geometria import generate_and_find_junctions
from .grafo import build_graph2, find_electrode_nodes2
from .fisica import build_admittance_matrix2


def setup_simulation(filepath: str = "config.ini",
                     parms: Optional[dict] = None) -> dict:
    """
    Setup a Neuromorphic simulation.

    Parameters
    ----------
    filepath : str
        Ruta al config.ini.
    parms : Optional[dict]
        Overrides sobre los parámetros del INI. Los valores derivados
        (PROXIMITY_THRESHOLD, R_WIRE_PER_LENGTH, TOTAL_TIME) se recalculan
        automáticamente tras el merge.

    Returns
    -------
    dict
        Diccionario con:
        - 'parameters': parámetros básicos y derivados
        - 'junctions': estructuras físicas de las junturas
        - 'graph': networkx.Graph
        - 'terminals': dict con 'input_nodes' y 'output_nodes'
        - 'circuit': matrices del circuito (Y, I, node_to_index)
    """
    simulation = cargar_parametros(filepath, parms=parms)

    generate_and_find_junctions(simulation)
    build_graph2(simulation)
    find_electrode_nodes2(simulation)
    build_admittance_matrix2(simulation)
    return simulation


if __name__ == "__main__":
    try:
        sim = setup_simulation()
        p = sim["parameters"]
        print("✅ ¡Archivo de configuración leído con éxito!")
        print(f"   NUM_WIRES = {p['NUM_WIRES']}")
        print(f"   R_WIRE_PER_LENGTH calculada = {p['R_WIRE_PER_LENGTH']:.4f} Ohm/um")
        print(f"   PROXIMITY_THRESHOLD = {p['PROXIMITY_THRESHOLD']} um")
        print(f"   Nodos de grafo: {sim['graph'].number_of_nodes()}")
        print(f"   Inputs: {len(sim['terminals']['input_nodes'])} | "
              f"Outputs: {len(sim['terminals']['output_nodes'])}")
    except Exception as e:
        print(f"❌ Error al cargar los parámetros: {e}")