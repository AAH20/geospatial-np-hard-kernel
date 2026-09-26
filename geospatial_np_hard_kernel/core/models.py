"""Data contracts and model definitions for the 10 Apex NP-Hard Problems in GIS and Geospatial Intelligence (GEOINT)."""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any

# --- Problem 1: Point-Feature Cartographic Label Placement (PALP) ---
@dataclass
class PointFeature:
    feature_id: str
    name: str
    x: float
    y: float
    priority: float
    width: float = 20.0
    height: float = 10.0

@dataclass
class LabelPlacementResult:
    placements: Dict[str, Tuple[float, float, float, float]]  # feature_id -> (xmin, ymin, xmax, ymax)
    total_labels_placed: int
    conflict_count: int
    aesthetic_score: float
    algorithm: str
    execution_time_us: float

# --- Problem 2: Capacitated p-Median & Emergency Facility Location (CPMP) ---
@dataclass
class DemandPoint:
    point_id: str
    x: float
    y: float
    demand_weight: float

@dataclass
class CandidateFacility:
    facility_id: str
    x: float
    y: float
    capacity: float
    fixed_cost: float = 100.0

@dataclass
class FacilityLocationResult:
    open_facilities: List[str]
    assignments: Dict[str, str]  # demand_id -> facility_id
    total_weighted_distance: float
    capacity_utilization_pct: float
    algorithm: str
    execution_time_us: float

# --- Problem 3: Art Gallery & Terrain Viewshed Radar Siting ---
@dataclass
class TerrainPoint:
    x: int
    y: int
    elevation: float

@dataclass
class ViewshedResult:
    selected_sensors: List[Tuple[int, int]]
    total_covered_cells: int
    coverage_percentage: float
    redundancy_overlap_ratio: float
    algorithm: str
    execution_time_us: float

# --- Problem 4: Spatial Contiguity-Constrained Territorial Districting ---
@dataclass
class SpatialTract:
    tract_id: str
    x: float
    y: float
    population: float
    neighbors: List[str]

@dataclass
class TerritorialDistrictingResult:
    districts: Dict[int, List[str]]  # district_id -> tract_ids
    population_disparity_pct: float
    contiguity_certified: bool
    compactness_score: float
    algorithm: str
    execution_time_us: float

# --- Problem 5: Dubins Multi-UAV Trajectory with Threat Domes (DTSPN) ---
@dataclass
class UAVWaypoint:
    wp_id: str
    x: float
    y: float
    desired_heading_deg: float = 0.0

@dataclass
class ThreatDome:
    center_x: float
    center_y: float
    radius: float
    lethality: float

@dataclass
class DubinsTrajectoryResult:
    waypoint_order: List[str]
    total_path_length: float
    threat_exposure_penalty: float
    kinematic_feasible: bool
    algorithm: str
    execution_time_us: float

# --- Problem 6: Homotopy-Preserving Cartographic Line Generalization ---
@dataclass
class SimplificationResult:
    simplified_points: List[Tuple[float, float]]
    vertex_reduction_pct: float
    max_frechet_deviation: float
    self_intersection_free: bool
    algorithm: str
    execution_time_us: float

# --- Problem 7: Time-Dependent Evacuation Route Scheduling (TD-VRPTW) ---
@dataclass
class EvacuationZone:
    zone_id: str
    population: int
    urgency: float

@dataclass
class RoadSegment:
    source: str
    target: str
    travel_time_base_min: float
    capacity_flow: float
    flood_delay_factor: float = 1.0

@dataclass
class EvacuationRouteResult:
    evacuation_paths: Dict[str, List[str]]
    clearance_time_min: float
    chokepoint_bottleneck_ratio: float
    algorithm: str
    execution_time_us: float

# --- Problem 8: Spatial Weights Matrix Optimization & Moran's I ---
@dataclass
class SpatialFeature:
    feature_id: str
    x: float
    y: float
    val: float

@dataclass
class SpatialWeightsResult:
    morans_i: float
    z_score: float
    p_value: float
    matrix_sparsity_pct: float
    algorithm: str
    execution_time_us: float

# --- Problem 9: Spatial Co-Location Pattern Mining ---
@dataclass
class SpatialInstance:
    instance_id: str
    feature_type: str
    x: float
    y: float

@dataclass
class CoLocationPatternResult:
    prevalent_patterns: List[Tuple[Tuple[str, ...], float]]  # (pattern, participation_index)
    total_patterns_found: int
    min_prevalence_threshold: float
    algorithm: str
    execution_time_us: float

# --- Problem 10: Agile Satellite Constellation Tasking (AEOSSP) ---
@dataclass
class ObservationTarget:
    target_id: str
    x: float
    y: float
    priority: float
    window_start: float
    window_end: float
    duration: float

@dataclass
class ConstellationScheduleResult:
    scheduled_tasks: List[Tuple[str, float, float]]  # (target_id, start_time, end_time)
    total_collected_priority: float
    capacity_utilization_pct: float
    slew_transition_certified: bool
    algorithm: str
    execution_time_us: float
