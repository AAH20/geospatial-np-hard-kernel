"""Urban Civilian GIS & Infrastructure Planning Adapter (ArcGIS / QGIS Compatible).

Provides end-to-end municipal workflows: automated cartographic typography placement,
emergency EMS / fire station facility location, civic districting, evacuation corridor routing,
cadastral polyline generalization, and spatial autocorrelation clustering.
"""
from typing import List, Dict, Tuple, Any
from ..core.models import (
    PointFeature,
    LabelPlacementResult,
    DemandPoint,
    CandidateFacility,
    FacilityLocationResult,
    SpatialTract,
    TerritorialDistrictingResult,
    EvacuationZone,
    RoadSegment,
    EvacuationRouteResult,
    SimplificationResult,
    SpatialFeature,
    SpatialWeightsResult,
)
from ..core.label_placement import solve_label_placement
from ..core.facility_location import solve_capacitated_p_median
from ..core.territorial_districting import solve_territorial_districting
from ..core.evacuation_routing import solve_evacuation_routing
from ..core.line_simplification import solve_line_simplification
from ..core.spatial_weights import solve_spatial_weights_optimization


class UrbanCivilianGISAdapter:
    """Specialized adapter for municipal GIS, cartography, and urban civil protection."""

    @staticmethod
    def render_cartographic_labels(
        map_features: List[PointFeature]
    ) -> LabelPlacementResult:
        """Automates zero-conflict cartographic label placement for city map production."""
        return solve_label_placement(map_features)

    @staticmethod
    def plan_emergency_medical_stations(
        neighborhood_demands: List[DemandPoint],
        candidate_depots: List[CandidateFacility],
        stations_to_open: int = 3
    ) -> FacilityLocationResult:
        """Determines optimal locations for ambulance stations minimizing response time under capacity caps."""
        return solve_capacitated_p_median(neighborhood_demands, candidate_depots, p=stations_to_open)

    @staticmethod
    def partition_municipal_wards(
        census_tracts: List[SpatialTract],
        ward_count: int = 3
    ) -> TerritorialDistrictingResult:
        """Partitions urban census tracts into contiguous, population-balanced voting or service wards."""
        return solve_territorial_districting(census_tracts, k_districts=ward_count)

    @staticmethod
    def schedule_civil_evacuation(
        zones: List[EvacuationZone],
        road_network: List[RoadSegment],
        safe_havens: List[str]
    ) -> EvacuationRouteResult:
        """Calculates dynamic evacuation routes and identifies bottleneck chokepoints during flood/wildfire."""
        return solve_evacuation_routing(zones, road_network, safe_havens)

    @staticmethod
    def generalize_shorelines_or_cadastre(
        polyline_vertices: List[Tuple[float, float]],
        tolerance_meters: float = 2.0
    ) -> SimplificationResult:
        """Simplifies complex vector boundaries without creating topological self-intersections."""
        return solve_line_simplification(polyline_vertices, epsilon=tolerance_meters)

    @staticmethod
    def detect_spatial_clustering(
        features: List[SpatialFeature]
    ) -> SpatialWeightsResult:
        """Finds optimal spatial weights matrix maximizing Moran's I spatial autocorrelation significance."""
        return solve_spatial_weights_optimization(features)
