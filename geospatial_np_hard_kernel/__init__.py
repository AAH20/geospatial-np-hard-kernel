"""Geospatial NP-Hard Kernel: Solvers for the 10 Apex NP-Hard Problems in GIS and Geospatial Intelligence."""
from .engine import GeospatialNPHardEngine
from .adapters.defense_geoint_surveillance import DefenseGEOINTAdapter
from .adapters.urban_civilian_gis import UrbanCivilianGISAdapter

__version__ = "1.0.0"
__all__ = [
    "GeospatialNPHardEngine",
    "DefenseGEOINTAdapter",
    "UrbanCivilianGISAdapter",
]
