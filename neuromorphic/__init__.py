from . import geometria
from . import grafo
from .fisica import update_stochastic_conductance2
from . import simulador
from .simulation import setup_simulation

__version__ = "0.1.0"

__all__ = [
    "setup_simulation",
    "update_stochastic_conductance2",
]
