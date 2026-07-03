import configparser
import numpy as np

def cargar_parametros(filepath="config.ini"):
    config = configparser.ConfigParser()
    config.read(filepath)
    
    params = {}
    
    # --- Red ---
    params['NUM_WIRES'] = config.getint('Network', 'num_wires')
    params['AREA'] = config.getfloat('Network', 'area')
    params['LENGTH'] = config.getfloat('Network', 'length')
    
    p_ratio = config.getfloat('Network', 'proximity_threshold_ratio')
    params['PROXIMITY_THRESHOLD'] = p_ratio * params['AREA']
    
    # --- Eléctricos y cálculo de R_WIRE_PER_LENGTH ---
    params['V_INPUT'] = config.getfloat('Electrical', 'v_input')
    params['V_GROUND'] = config.getfloat('Electrical', 'v_ground')
    
    rho = config.getfloat('Electrical', 'rho_plata')
    diametro = config.getfloat('Electrical', 'diametro_nm')
    radio_um = (diametro / 2) * 1e-3
    params['R_WIRE_PER_LENGTH'] = rho / (np.pi * (radio_um ** 2))
    
    # --- Tiempo ---
    params['TOTAL_TIME'] = config.getfloat('Time', 'total_time')
    params['TIME_STEP_DT'] = config.getfloat('Time', 'time_step_dt')
    
    # --- Memristor ---
    params['G_ON'] = config.getfloat('Memristor', 'g_on')
    params['G_OFF'] = config.getfloat('Memristor', 'g_off')
    params['V_THRESHOLD'] = config.getfloat('Memristor', 'v_threshold')
    
    # --- Probabilidades ---
    params['P0_SET'] = config.getfloat('Probabilities', 'p0_set')
    params['ALPHA_SET'] = config.getfloat('Probabilities', 'alpha_set')
    params['P_DECAY'] = config.getfloat('Probabilities', 'p_decay')
    params['BETA_RESET'] = config.getfloat('Probabilities', 'beta_reset')
    
    return params