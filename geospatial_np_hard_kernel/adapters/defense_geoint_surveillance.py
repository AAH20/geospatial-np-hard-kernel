"""Defense Geospatial Intelligence (GEOINT) and Multi-Domain Surveillance Adapter.

Applies NP-hard geospatial solvers to tactical multi-UAV path planning through threat domes,
border radar viewshed coverage, agile satellite constellation reconnaissance tasking,
and SIGINT/ELINT spatial co-location detection.
"""
from typing import List, Dict, Tuple, Any
from ..core.models import (
    UAVWaypoint,
    ThreatDome,
    DubinsTrajectoryResult,
    TerrainPoint,
    ViewshedResult,
    ObservationTarget,
    ConstellationScheduleResult,
    SpatialInstance,
    CoLocationPatternResult,
)
from ..core.dubins_uav_path import solve_dubins_uav_path
from ..core.viewshed_siting import solve_viewshed_siting
from ..core.satellite_tasking import solve_satellite_tasking
from ..core.colocation_mining import solve_colocation_mining


class DefenseGEOINTAdapter:
    """Specialized adapter for defense geospatial intelligence, ISR, and multi-domain operations."""

    @staticmethod
    def plan_stealth_recon_flight(
        waypoints: List[UAVWaypoint],
        threat_domes: List[ThreatDome],
        turn_radius: float = 8.0
    ) -> DubinsTrajectoryResult:
        """Plans optimal UAV trajectory navigating through SAM threat domes."""
        return solve_dubins_uav_path(waypoints, threat_domes, min_turn_radius=turn_radius)

    @staticmethod
    def deploy_border_radar_network(
        dem_points: List[TerrainPoint],
        candidate_radar_posts: List[Tuple[int, int]],
        radar_range: float = 15.0,
        radar_units: int = 4,
        mast_height: float = 5.0
    ) -> ViewshedResult:
        """Optimizes air-defense or border surveillance radar siting over complex terrain."""
        return solve_viewshed_siting(
            dem_points,
            candidate_radar_posts,
            sensor_range=radar_range,
            k_sensors=radar_units,
            sensor_mast_height=mast_height
        )

    @staticmethod
    def task_recon_constellation(
        recon_targets: List[ObservationTarget],
        slew_rate_deg_per_sec: float = 4.0,
        orbit_duration: float = 1200.0
    ) -> ConstellationScheduleResult:
        """Schedules high-priority agile satellite imagery acquisitions with attitude slew feasibility."""
        return solve_satellite_tasking(
            recon_targets,
            slew_rate_deg_per_sec=slew_rate_deg_per_sec,
            mission_duration=orbit_duration
        )

    @staticmethod
    def mine_adversarial_signatures(
        sigint_instances: List[SpatialInstance],
        proximity_threshold: float = 12.0,
        min_prevalence: float = 0.5
    ) -> CoLocationPatternResult:
        """Identifies co-located electronic warfare, missile launcher, and radar signatures."""
        return solve_colocation_mining(
            sigint_instances,
            distance_threshold=proximity_threshold,
            min_prevalence=min_prevalence
        )
