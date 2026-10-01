# load_config.py
import configparser
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parent


def cargar_parametros(filepath="config.ini"):
    path = Path(filepath)
    if not path.is_absolute() and not path.exists():
        candidate = PACKAGE_DIR / filepath
        if candidate.exists():
            filepath = str(candidate)

    config = configparser.ConfigParser()
    config.read(filepath)

    params = {}

    # --- Red de nanohilos ---
    params['NUM_WIRES'] = config.getint('Network', 'num_wires')
    params['AREA'] = config.getfloat('Network', 'area')
    params['LENGTH'] = config.getfloat('Network', 'length')

    p_ratio = config.getfloat('Network', 'proximity_threshold_ratio')
    params['PROXIMITY_THRESHOLD'] = p_ratio * params['AREA']

    # --- Parámetros eléctricos ---
    params['V_INPUT'] = config.getfloat('Electrical', 'v_input')
    params['V_GROUND'] = config.getfloat('Electrical', 'v_ground')
    params['V_READ'] = config.getfloat('Electrical', 'v_read')

    rho = config.getfloat('Electrical', 'rho_plata')
    diametro = config.getfloat('Electrical', 'diametro_nm')
    radio_um = (diametro / 2.0) * 1e-3
    params['R_WIRE_PER_LENGTH'] = rho / (np.pi * (radio_um ** 2))

    # --- Bucle temporal ---
    params['TOTAL_TIME'] = config.getfloat('Time', 'total_time')
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

    return {"parameters": params}


if __name__ == "__main__":
    try:
        sim = cargar_parametros()
        p = sim["parameters"]
        print("✅ ¡Archivo de configuración leído con éxito!")
        print(f"   NUM_WIRES = {p['NUM_WIRES']}")
        print(f"   R_WIRE_PER_LENGTH calculada = {p['R_WIRE_PER_LENGTH']:.4f} Ohm/um")
        print(f"   PROXIMITY_THRESHOLD = {p['PROXIMITY_THRESHOLD']} um")
        print(f"   T_PULSE = {p['T_PULSE']} s | T_RELAX = {p['T_RELAX']} s | V_READ = {p['V_READ']} V")
    except Exception as e:
        print(f"❌ Error al cargar los parámetros: {e}")