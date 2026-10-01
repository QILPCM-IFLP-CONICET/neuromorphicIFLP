import pytest

from neuromorphic.load_config import (
    KNOWN_RAW_KEYS,
    cargar_parametros,
)


def test_defaults_load():
    """Loading without arguments uses packaged defaults."""
    result = cargar_parametros()
    p = result["parameters"]
    assert p["NUM_WIRES"] > 0
    assert p["AREA"] > 0
    assert p["LENGTH"] > 0


def test_overrides_applied():
    result = cargar_parametros(parms={"NUM_WIRES": 42})
    assert result["parameters"]["NUM_WIRES"] == 42


def test_derived_recomputed_after_override():
    """AREA override must propagate to PROXIMITY_THRESHOLD."""
    result = cargar_parametros(parms={"AREA": 500.0})
    p = result["parameters"]
    assert p["PROXIMITY_THRESHOLD"] == p["PROXIMITY_THRESHOLD_RATIO"] * 500.0


def test_total_time_derived():
    result = cargar_parametros(parms={"T_PULSE": 3.0, "T_RELAX": 7.0})
    assert result["parameters"]["TOTAL_TIME"] == 10.0


def test_unknown_parameter_raises():
    with pytest.raises(KeyError, match="NUM_WIRE"):  # typo intentional
        cargar_parametros(parms={"NUM_WIRE": 100})


def test_explicit_filepath_missing_raises(tmp_path):
    missing = tmp_path / "does_not_exist.ini"
    with pytest.raises(FileNotFoundError):
        cargar_parametros(filepath=missing)


def test_known_keys_are_consistent():
    """Every known raw key must appear in the default-loaded params."""
    result = cargar_parametros()
    params = result["parameters"]
    missing = KNOWN_RAW_KEYS - params.keys()
    assert not missing, f"Known keys missing from defaults: {missing}"
