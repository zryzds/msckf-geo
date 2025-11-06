"""
GEOVINS: Geographic Visual-Inertial Navigation System

A comprehensive state estimation framework combining MSCKF with geographic constraints.
"""

__version__ = "0.1.0"

from . import core
from . import sensors
from . import features
from . import filter
from . import geographic

__all__ = ['core', 'sensors', 'features', 'filter', 'geographic']
