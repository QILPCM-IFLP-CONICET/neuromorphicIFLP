# load_config.py
import configparser
from pathlib import Path
from typing import Optional

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent


def _recompute_derived(params: dict) -> None:
    '''
    Calcula los valores derivados a partir de los valores crudos en params.

    Derivados:
    - PROXIMITY_THRESHOLD (um)  <- PROXIMITY_THRESHOLD_RATIO * AREA
    - R_WIRE_PER_LENGTH (Ohm/um) <- RHO_PLATA / (pi * radio_um^2)
    - TOTAL_TIME (s) <- T_PULSE + T_RELAX

    Llamar SOLO después de aplicar todos los overrides.
    '''
    params['PROXIMITY_THRESHOLD'] = (
        params['PROXIMITY_THRESHOLD_RATIO'] * params['AREA']
    )

    radio_um = (params['DIAMETRO_NM'] / 2.0) * 1e-3  # nm -> µm
    params['R_WIRE_PER_LENGTH'] = (
        params['RHO_PLATA'] / (np.pi * (radio_um ** 2))
    )

    params['TOTAL_TIME'] = params['T_PULSE'] + params['T_RELAX']


def cargar_parametros(filepath: str = "config.ini",
                      parms: Optional[dict] = None) -> dict:
    '''
    Lee un archivo de configuración INI, aplica overrides opcionales y
    calcula los valores derivados.

    Parameters
    ----------
    filepath : str
        Ruta al config.ini (relativa a CWD o al paquete).
    parms : Optional[dict]
        Overrides que pisan los valores del INI. Las claves deben coincidir
        con los nombres expuestos en params (ver más abajo). Los valores
        derivados se recalculan automáticamente después del merge.

    Returns
    -------
    dict
        {"parameters": {...}} con todos los parámetros listos para usar.
    '''
    path = Path(filepath)
    if not path.is_absolute() and not path.exists():
        candidate = PACKAGE_DIR / filepath
        if candidate.exists():
            filepath = str(candidate)

    config = configparser.ConfigParser()
    config.read(filepath)

    params = {}

    # --- Red de nanohilos (crudos) ---
    params['NUM_WIRES'] = config.getint('Network', 'num_wires')
    params['AREA'] = config.getfloat('Network', 'area')
    params['LENGTH'] = config.getfloat('Network', 'length')
    params['PROXIMITY_THRESHOLD_RATIO'] = config.getfloat(
        'Network', 'proximity_threshold_ratio'
    )

    # --- Parámetros eléctricos (crudos) ---
    params['V_INPUT'] = config.getfloat('Electrical', 'v_input')
    params['V_GROUND'] = config.getfloat('Electrical', 'v_ground')
    params['V_READ'] = config.getfloat('Electrical', 'v_read')
    params['RHO_PLATA'] = config.getfloat('Electrical', 'rho_plata')
    params['DIAMETRO_NM'] = config.getfloat('Electrical', 'diametro_nm')

    # --- Bucle temporal ---
    params['TIME_STEP_DT'] = config.getfloat('Time', 'time_step_dt')
    params['T_PULSE'] = config.getfloat('Time', 't_pulse')
    params['T_RELAX'] = config.getfloat('Time', 't_relax')

    # --- Memristor ---
    params['G_ON'] = config.getfloat('Memristor', 'g_on')
    params['G_OFF'] = config.getfloat('Memristor', 'g_off')
    params['V_THRESHOLD'] = config.getfloat('Memristor', 'v_threshold')

    # --- Probabilidades ---
    params['P0_SET'] = config.getfloat('Probabilities', 'p0_set')
    params['ALPHA_SET'] = config.getfloat('Probabilities', 'alpha_set')
    params['P_DECAY'] = config.getfloat('Probabilities', 'p_decay')
    params['BETA_RESET'] = config.getfloat('Probabilities', 'beta_reset')

    # --- Aplicar overrides ANTES de calcular derivados ---
    if parms is not None:
        params.update(parms)

    # --- Calcular derivados una sola vez, sobre el estado final ---
    _recompute_derived(params)

    return {"parameters": params}


if __name__ == "__main__":
    try:
        sim = cargar_parametros()
        p = sim["parameters"]
        print("✅ ¡Archivo de configuración leído con éxito!")
        print(f"   NUM_WIRES = {p['NUM_WIRES']}")
        print(f"   R_WIRE_PER_LENGTH calculada = {p['R_WIRE_PER_LENGTH']:.4f} Ohm/um")
        print(f"   PROXIMITY_THRESHOLD = {p['PROXIMITY_THRESHOLD']} um")
        print(f"   T_PULSE = {p['T_PULSE']} s | T_RELAX = {p['T_RELAX']} s "
              f"| V_READ = {p['V_READ']} V")
        print(f"   TOTAL_TIME (derivado) = {p['TOTAL_TIME']} s")
    except Exception as e:
        print(f"❌ Error al cargar los parámetros: {e}")