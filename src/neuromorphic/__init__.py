"""Neuromorphic nanowire network simulator."""

from .dinamica import run_simulation_dynamic_pulse
from .fisica import EVOLVE_MODELS
from .simulation import setup_simulation
from .visualizacion import plot_simulation_results

__version__ = "0.1.0"

__all__ = [
    "EVOLVE_MODELS",
    "plot_simulation_results",
    "run_simulation_dynamic_pulse",
    "setup_simulation",
]
