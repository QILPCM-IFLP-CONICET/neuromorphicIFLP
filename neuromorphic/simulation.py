from typing import Optional

from .load_config import cargar_parametros
from .geometria import generate_and_find_junctions
from .grafo import build_graph2, find_electrode_nodes2
from .fisica import build_admittance_matrix2

def setup_simulation(filepath:str="config.ini" ,parms:Optional[dict]=None)->dict:
    """
    Setup a Neuromorphic simulation.
    
    Parameters
    ----------
    
    filepath: str 
        path to an init file that defines the default parameters.
    
    parms: Optional[dict]
        dict of parameters that overwrite the default parameters in the
        config.ini file.

    Return 
    ------

    dict:
        a Python dict entries defining different aspects of the simulation:

        - 'parameters': the basic parameters
        - 'junctions': dict of structures describing the physical junctions.
        - 'graph': the networkx Graph object defining the connectivity of the nodes.
        - 'terminals': dict of list of input and output nodes. 

    """
    simulation= cargar_parametros(filepath)
    if parms is not None:
        simulation["parameters"].update(parms)

    generate_and_find_junctions(simulation)
    build_graph2(simulation)
    find_electrode_nodes2(simulation)
    build_admittance_matrix2(simulation)
    return simulation


# Bloque de prueba para verificar que lee bien el config.ini
if __name__ == "__main__":
    try:
        sim = setup_simulation()
        print("✅ ¡Archivo de configuración leído con éxito!")
        print(f"   NUM_WIRES = {p['NUM_WIRES']}")
        print(f"   R_WIRE_PER_LENGTH calculada = {p['R_WIRE_PER_LENGTH']:.4f} Ohm/um")
        print(f"   PROXIMITY_THRESHOLD = {p['PROXIMITY_THRESHOLD']} um")
    except Exception as e:
        print(f"❌ Error al cargar los parámetros: {e}")
