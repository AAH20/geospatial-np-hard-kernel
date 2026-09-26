"""Unified Engine for the 10 Apex NP-Hard Problems in GIS and Geospatial Intelligence."""
import math
import time
from typing import Dict, Any, List, Tuple

from .core.models import (
    PointFeature,
    DemandPoint,
    CandidateFacility,
    TerrainPoint,
    SpatialTract,
    UAVWaypoint,
    ThreatDome,
    EvacuationZone,
    RoadSegment,
    SpatialFeature,
    SpatialInstance,
    ObservationTarget,
)
from .core.label_placement import solve_label_placement
from .core.facility_location import solve_capacitated_p_median
from .core.viewshed_siting import solve_viewshed_siting
from .core.territorial_districting import solve_territorial_districting
from .core.dubins_uav_path import solve_dubins_uav_path
from .core.line_simplification import solve_line_simplification
from .core.evacuation_routing import solve_evacuation_routing
from .core.spatial_weights import solve_spatial_weights_optimization
from .core.colocation_mining import solve_colocation_mining
from .core.satellite_tasking import solve_satellite_tasking

from .adapters.defense_geoint_surveillance import DefenseGEOINTAdapter
from .adapters.urban_civilian_gis import UrbanCivilianGISAdapter


class GeospatialNPHardEngine:
    """Unified engine coordinating NP-hard solvers across GIS and Geospatial Intelligence."""

    def __init__(self):
        self.defense_adapter = DefenseGEOINTAdapter()
        self.urban_adapter = UrbanCivilianGISAdapter()

    # --- Core Direct Solvers ---
    solve_palp = staticmethod(solve_label_placement)
    solve_cpmp = staticmethod(solve_capacitated_p_median)
    solve_viewshed = staticmethod(solve_viewshed_siting)
    solve_districting = staticmethod(solve_territorial_districting)
    solve_dtspn = staticmethod(solve_dubins_uav_path)
    solve_min_vertex = staticmethod(solve_line_simplification)
    solve_evacuation = staticmethod(solve_evacuation_routing)
    solve_morans_w = staticmethod(solve_spatial_weights_optimization)
    solve_colocation = staticmethod(solve_colocation_mining)
    solve_aeossp = staticmethod(solve_satellite_tasking)

    # --- Benchmark Synthesis ---
    @staticmethod
    def generate_synthetic_benchmark_suite() -> Dict[str, Any]:
        """Generates rigorous synthetic benchmark problem instances for all 10 NP-hard challenges."""
        # 1. PALP features
        palp_features = [
            PointFeature(f"pt_{i}", f"City_{i}", (i * 27) % 150, (i * 43) % 150, priority=100.0 - i * 5)
            for i in range(15)
        ]

        # 2. CPMP demand and facilities
        demands = [
            DemandPoint(f"d_{i}", (i * 19) % 100, (i * 31) % 100, demand_weight=10.0 + (i % 5) * 5)
            for i in range(20)
        ]
        facilities = [
            CandidateFacility(f"fac_{j}", j * 22.0, j * 20.0, capacity=120.0)
            for j in range(5)
        ]

        # 3. Viewshed terrain grid (12x12)
        grid = []
        for x in range(12):
            for y in range(12):
                elev = 10.0 * math.sin(x * 0.5) + 8.0 * math.cos(y * 0.5)
                grid.append(TerrainPoint(x, y, round(elev, 2)))
        radars = [(2, 2), (2, 9), (9, 2), (9, 9), (5, 5)]

        # 4. Districting tracts (12 tracts on a grid with known adjacency)
        tracts = []
        for row in range(3):
            for col in range(4):
                tid = f"t_{row}_{col}"
                nbrs = []
                if row > 0: nbrs.append(f"t_{row-1}_{col}")
                if row < 2: nbrs.append(f"t_{row+1}_{col}")
                if col > 0: nbrs.append(f"t_{row}_{col-1}")
                if col < 3: nbrs.append(f"t_{row}_{col+1}")
                pop = 1000.0 + (row * 4 + col) * 50.0
                tracts.append(SpatialTract(tid, col * 10.0, row * 10.0, pop, nbrs))

        # 5. DTSPN UAV waypoints & threat domes
        uav_wps = [
            UAVWaypoint(f"wp_{i}", 30.0 * math.cos(i * 0.8), 30.0 * math.sin(i * 0.8), desired_heading_deg=i * 45.0)
            for i in range(8)
        ]
        threats = [
            ThreatDome(10.0, 10.0, radius=12.0, lethality=5.0),
            ThreatDome(-15.0, -10.0, radius=10.0, lethality=4.0)
        ]

        # 6. Line generalization polyline (25 vertices with sinusoidal curve)
        line_pts = [(float(i), 5.0 * math.sin(i * 0.4) + (i % 3) * 0.5) for i in range(25)]

        # 7. Evacuation zones & roads
        evac_zones = [
            EvacuationZone(f"zone_{i}", population=500 + i * 150, urgency=float(i + 1))
            for i in range(5)
        ]
        roads = [
            RoadSegment("zone_0", "junc_1", travel_time_base_min=5.0, capacity_flow=400.0),
            RoadSegment("zone_1", "junc_1", travel_time_base_min=4.0, capacity_flow=350.0),
            RoadSegment("zone_2", "junc_2", travel_time_base_min=6.0, capacity_flow=500.0),
            RoadSegment("zone_3", "junc_2", travel_time_base_min=3.0, capacity_flow=300.0),
            RoadSegment("zone_4", "junc_1", travel_time_base_min=7.0, capacity_flow=450.0),
            RoadSegment("junc_1", "safe_A", travel_time_base_min=10.0, capacity_flow=800.0, flood_delay_factor=1.2),
            RoadSegment("junc_2", "safe_B", travel_time_base_min=8.0, capacity_flow=900.0, flood_delay_factor=1.1),
            RoadSegment("junc_1", "junc_2", travel_time_base_min=3.0, capacity_flow=200.0),
        ]
        safe_havens = ["safe_A", "safe_B"]

        # 8. Spatial weights features (16 features in 2 spatial clusters)
        spat_feats = []
        for i in range(8):
            spat_feats.append(SpatialFeature(f"c1_{i}", 5.0 + (i % 3), 5.0 + (i // 3), val=50.0 + i))
            spat_feats.append(SpatialFeature(f"c2_{i}", 40.0 + (i % 3), 40.0 + (i // 3), val=10.0 + i))

        # 9. Co-location instances (types A, B, C clustered together)
        colo_insts = []
        for i in range(8):
            base_x = 20.0 * (i % 3)
            base_y = 20.0 * (i // 3)
            colo_insts.append(SpatialInstance(f"A_{i}", "Radar", base_x + 1.0, base_y + 1.0))
            colo_insts.append(SpatialInstance(f"B_{i}", "SAM_Battery", base_x + 2.0, base_y + 1.5))
            colo_insts.append(SpatialInstance(f"C_{i}", "CommandPost", base_x + 1.8, base_y + 2.2))

        # 10. Satellite observation targets
        sat_targets = [
            ObservationTarget(f"tgt_{i}", 10.0 * i, 15.0 * (i % 4), priority=20.0 + i * 10,
                              window_start=i * 25.0, window_end=i * 25.0 + 80.0, duration=15.0)
            for i in range(8)
        ]

        return {
            "palp": palp_features,
            "cpmp": (demands, facilities, 3),
            "viewshed": (grid, radars, 8.0, 3),
            "districting": (tracts, 3),
            "dtspn": (uav_wps, threats, 5.0),
            "simplification": (line_pts, 1.5),
            "evacuation": (evac_zones, roads, safe_havens),
            "morans_w": (spat_feats, [2, 3, 4]),
            "colocation": (colo_insts, 5.0, 0.4),
            "satellite": (sat_targets, 3.0, 500.0),
        }

    def benchmark_all_10(self) -> Dict[str, Any]:
        """Executes and benchmarks all 10 NP-hard GIS/GEOINT solvers in a single run."""
        suite = self.generate_synthetic_benchmark_suite()
        t_start_total = time.perf_counter()

        res_palp = self.solve_palp(suite["palp"])
        res_cpmp = self.solve_cpmp(*suite["cpmp"])
        res_viewshed = self.solve_viewshed(*suite["viewshed"])
        res_districting = self.solve_districting(*suite["districting"])
        res_dtspn = self.solve_dtspn(*suite["dtspn"])
        res_simplification = self.solve_min_vertex(*suite["simplification"])
        res_evacuation = self.solve_evacuation(*suite["evacuation"])
        res_morans = self.solve_morans_w(*suite["morans_w"])
        res_colocation = self.solve_colocation(*suite["colocation"])
        res_satellite = self.solve_aeossp(*suite["satellite"])

        t_end_total = time.perf_counter()
        total_time_us = (t_end_total - t_start_total) * 1_000_000.0

        return {
            "total_benchmark_time_us": round(total_time_us, 2),
            "solvers": {
                "P1_PALP_Label_Placement": {
                    "labels_placed": res_palp.total_labels_placed,
                    "aesthetic_score": res_palp.aesthetic_score,
                    "time_us": res_palp.execution_time_us,
                },
                "P2_CPMP_Facility_Location": {
                    "open_facilities": res_cpmp.open_facilities,
                    "utilization_pct": res_cpmp.capacity_utilization_pct,
                    "time_us": res_cpmp.execution_time_us,
                },
                "P3_Terrain_Viewshed_Siting": {
                    "covered_cells": res_viewshed.total_covered_cells,
                    "coverage_pct": res_viewshed.coverage_percentage,
                    "time_us": res_viewshed.execution_time_us,
                },
                "P4_Territorial_Districting": {
                    "population_disparity_pct": res_districting.population_disparity_pct,
                    "contiguity_certified": res_districting.contiguity_certified,
                    "time_us": res_districting.execution_time_us,
                },
                "P5_Dubins_UAV_Trajectory": {
                    "path_length": res_dtspn.total_path_length,
                    "threat_exposure": res_dtspn.threat_exposure_penalty,
                    "time_us": res_dtspn.execution_time_us,
                },
                "P6_Line_Generalization": {
                    "vertex_reduction_pct": res_simplification.vertex_reduction_pct,
                    "max_deviation": res_simplification.max_frechet_deviation,
                    "time_us": res_simplification.execution_time_us,
                },
                "P7_Evacuation_Routing": {
                    "clearance_time_min": res_evacuation.clearance_time_min,
                    "chokepoint_ratio": res_evacuation.chokepoint_bottleneck_ratio,
                    "time_us": res_evacuation.execution_time_us,
                },
                "P8_Spatial_Weights_Morans_I": {
                    "morans_i": res_morans.morans_i,
                    "z_score": res_morans.z_score,
                    "time_us": res_morans.execution_time_us,
                },
                "P9_Spatial_CoLocation_Mining": {
                    "patterns_found": res_colocation.total_patterns_found,
                    "time_us": res_colocation.execution_time_us,
                },
                "P10_Agile_Satellite_Tasking": {
                    "priority_collected": res_satellite.total_collected_priority,
                    "slew_certified": res_satellite.slew_transition_certified,
                    "time_us": res_satellite.execution_time_us,
                },
            }
        }
