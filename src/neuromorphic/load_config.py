# load_config.py
"""Configuration loading for the neuromorphic nanowire simulator.

Design
------
- ``_load_defaults()`` reads the ``.ini`` bundled with the package.
- ``_load_ini(path)`` reads an explicit user-provided ``.ini``.
- ``cargar_parametros(filepath=None, parms=None)`` is the single public
  entry point. It never searches the current working directory
  implicitly: if ``filepath`` is None, only the package defaults are used.

Unknown keys in ``parms`` trigger an ``UnknownParameterWarning`` and are
ignored (they do not override anything).
"""
from __future__ import annotations

import configparser
import warnings
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_INI_NAME = "defaults.ini"


class UnknownParameterWarning(UserWarning):
    """Emitted when ``parms`` contains keys not present in the loaded file."""


# Keys exposed in the parameter dict. Kept explicit so that typos in
# ``parms`` are surfaced as warnings rather than silently ignored.
KNOWN_RAW_KEYS: frozenset[str] = frozenset({
    # Network
    "NUM_WIRES", "AREA", "LENGTH", "PROXIMITY_THRESHOLD_RATIO",
    # Electrical
    "V_INPUT", "V_GROUND", "V_READ", "RHO_PLATA", "DIAMETRO_NM",
    # Time
    "TIME_STEP_DT", "T_PULSE", "T_RELAX",
    # Memristor
    "G_ON", "G_OFF", "V_THRESHOLD",
    # Probabilities
    "P0_SET", "ALPHA_SET", "P_DECAY", "BETA_RESET",
})

DERIVED_KEYS: frozenset[str] = frozenset({
    "PROXIMITY_THRESHOLD",
    "R_WIRE_PER_LENGTH",
    "TOTAL_TIME",
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def cargar_parametros(
    filepath: str | Path | None = None,
    parms: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Load simulation parameters.

    Parameters
    ----------
    filepath : str or Path, optional
        Path to an explicit ``.ini`` file. If ``None`` (default), the
        packaged ``defaults.ini`` is used. No silent CWD fallback.
    parms : dict, optional
        Overrides applied on top of the loaded file. Unknown keys emit
        ``UnknownParameterWarning`` and are ignored.

    Returns
    -------
    dict
        ``{"parameters": {...}}`` with raw and derived values.
    """

    params = _load_defaults() if filepath is None else _load_ini(Path(filepath))

    if parms is not None:
        unknown = set(parms) - KNOWN_RAW_KEYS
        if unknown:
            warnings.warn(
                f"Ignoring unknown parameter(s): {sorted(unknown)}. "
                f"Known raw keys: {sorted(KNOWN_RAW_KEYS)}",
                UnknownParameterWarning,
                stacklevel=2,
            )
        accepted = {k: v for k, v in parms.items() if k in KNOWN_RAW_KEYS}
        params.update(accepted)

    _recompute_derived(params)
    return {"parameters": params}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _load_defaults() -> dict[str, Any]:
    """Read the packaged ``defaults.ini`` via importlib.resources."""
    ini = files("neuromorphic").joinpath(DEFAULT_INI_NAME)
    if not ini.is_file():
        raise FileNotFoundError(
            f"Packaged {DEFAULT_INI_NAME} not found. "
            "The package is likely broken; reinstall it."
        )
    with ini.open("r", encoding="utf-8") as fh:
        return _parse_ini(fh)


def _load_ini(path: Path) -> dict[str, Any]:
    """Read a user-provided ``.ini`` from an explicit path."""
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return _parse_ini(fh)


def _parse_ini(fh) -> dict[str, Any]:
    config = configparser.ConfigParser()
    config.read_file(fh)

    params: dict[str, Any] = {}

    # Network
    params["NUM_WIRES"] = config.getint("Network", "num_wires")
    params["AREA"] = config.getfloat("Network", "area")
    params["LENGTH"] = config.getfloat("Network", "length")
    params["PROXIMITY_THRESHOLD_RATIO"] = config.getfloat(
        "Network", "proximity_threshold_ratio"
    )

    # Electrical
    params["V_INPUT"] = config.getfloat("Electrical", "v_input")
    params["V_GROUND"] = config.getfloat("Electrical", "v_ground")
    params["V_READ"] = config.getfloat("Electrical", "v_read")
    params["RHO_PLATA"] = config.getfloat("Electrical", "rho_plata")
    params["DIAMETRO_NM"] = config.getfloat("Electrical", "diametro_nm")

    # Time
    params["TIME_STEP_DT"] = config.getfloat("Time", "time_step_dt")
    params["T_PULSE"] = config.getfloat("Time", "t_pulse")
    params["T_RELAX"] = config.getfloat("Time", "t_relax")

    # Memristor
    params["G_ON"] = config.getfloat("Memristor", "g_on")
    params["G_OFF"] = config.getfloat("Memristor", "g_off")
    params["V_THRESHOLD"] = config.getfloat("Memristor", "v_threshold")

    # Probabilities
    params["P0_SET"] = config.getfloat("Probabilities", "p0_set")
    params["ALPHA_SET"] = config.getfloat("Probabilities", "alpha_set")
    params["P_DECAY"] = config.getfloat("Probabilities", "p_decay")
    params["BETA_RESET"] = config.getfloat("Probabilities", "beta_reset")

    return params


def _recompute_derived(params: dict[str, Any]) -> None:
    """Recalculate derived values in-place from raw keys."""
    params["PROXIMITY_THRESHOLD"] = (
        params["PROXIMITY_THRESHOLD_RATIO"] * params["AREA"]
    )
    radio_um = (params["DIAMETRO_NM"] / 2.0) * 1e-3
    params["R_WIRE_PER_LENGTH"] = (
        params["RHO_PLATA"] / (np.pi * radio_um**2)
    )
    params["TOTAL_TIME"] = params["T_PULSE"] + params["T_RELAX"]
