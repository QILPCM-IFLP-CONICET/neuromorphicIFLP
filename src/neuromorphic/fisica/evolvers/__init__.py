"""
Base module for defining and register memristor evolution models.
"""

# Importar los módulos registra los modelos (numba se carga recién al usarlo)
from . import ladder, ladder_numba, thermal  # noqa: F401
from .base import (
    EVOLVE_MODELS,
    EVOLVER_SPECS,
    EvolverSpec,
    declared_evolver_parameters,
    init_all_off,
    initialize_evolver,
    register_memristor_evol_model,
    set_all_memristors,
)

__all__ = [
    "EVOLVER_SPECS",
    "EVOLVE_MODELS",
    "EvolverSpec",
    "declared_evolver_parameters",
    "init_all_off",
    "initialize_evolver",
    "register_memristor_evol_model",
    "set_all_memristors",
]
