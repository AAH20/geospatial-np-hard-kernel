"""Core algorithms and models for the 10 Apex NP-Hard Problems in GIS and Geospatial Intelligence."""
from .models import (
    PointFeature,
    LabelPlacementResult,
    DemandPoint,
    CandidateFacility,
    FacilityLocationResult,
    TerrainPoint,
    ViewshedResult,
    SpatialTract,
    TerritorialDistrictingResult,
    UAVWaypoint,
    ThreatDome,
    DubinsTrajectoryResult,
    SimplificationResult,
    EvacuationZone,
    RoadSegment,
    EvacuationRouteResult,
    SpatialFeature,
    SpatialWeightsResult,
    SpatialInstance,
    CoLocationPatternResult,
    ObservationTarget,
    ConstellationScheduleResult,
)

from .label_placement import solve_label_placement
from .facility_location import solve_capacitated_p_median
from .viewshed_siting import solve_viewshed_siting
from .territorial_districting import solve_territorial_districting
from .dubins_uav_path import solve_dubins_uav_path
from .line_simplification import solve_line_simplification
from .evacuation_routing import solve_evacuation_routing
from .spatial_weights import solve_spatial_weights_optimization
from .colocation_mining import solve_colocation_mining
from .satellite_tasking import solve_satellite_tasking

__all__ = [
    # Models
    "PointFeature",
    "LabelPlacementResult",
    "DemandPoint",
    "CandidateFacility",
    "FacilityLocationResult",
    "TerrainPoint",
    "ViewshedResult",
    "SpatialTract",
    "TerritorialDistrictingResult",
    "UAVWaypoint",
    "ThreatDome",
    "DubinsTrajectoryResult",
    "SimplificationResult",
    "EvacuationZone",
    "RoadSegment",
    "EvacuationRouteResult",
    "SpatialFeature",
    "SpatialWeightsResult",
    "SpatialInstance",
    "CoLocationPatternResult",
    "ObservationTarget",
    "ConstellationScheduleResult",
    # Solvers
    "solve_label_placement",
    "solve_capacitated_p_median",
    "solve_viewshed_siting",
    "solve_territorial_districting",
    "solve_dubins_uav_path",
    "solve_line_simplification",
    "solve_evacuation_routing",
    "solve_spatial_weights_optimization",
    "solve_colocation_mining",
    "solve_satellite_tasking",
]
