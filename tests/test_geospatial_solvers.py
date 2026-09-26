"""Comprehensive Test Suite for all 10 Apex GIS & GEOINT NP-Hard Solvers."""
import unittest
import math
from geospatial_np_hard_kernel.core.models import (
    PointFeature, DemandPoint, CandidateFacility, TerrainPoint, SpatialTract,
    UAVWaypoint, ThreatDome, EvacuationZone, RoadSegment, SpatialFeature,
    SpatialInstance, ObservationTarget
)
from geospatial_np_hard_kernel.core.label_placement import solve_label_placement
from geospatial_np_hard_kernel.core.facility_location import solve_capacitated_p_median
from geospatial_np_hard_kernel.core.viewshed_siting import solve_viewshed_siting
from geospatial_np_hard_kernel.core.territorial_districting import solve_territorial_districting
from geospatial_np_hard_kernel.core.dubins_uav_path import solve_dubins_uav_path
from geospatial_np_hard_kernel.core.line_simplification import solve_line_simplification
from geospatial_np_hard_kernel.core.evacuation_routing import solve_evacuation_routing
from geospatial_np_hard_kernel.core.spatial_weights import solve_spatial_weights_optimization
from geospatial_np_hard_kernel.core.colocation_mining import solve_colocation_mining
from geospatial_np_hard_kernel.core.satellite_tasking import solve_satellite_tasking
from geospatial_np_hard_kernel.engine import GeospatialNPHardEngine


class TestGeospatialNPHardKernel(unittest.TestCase):

    def test_p1_cartographic_label_placement(self):
        features = [
            PointFeature("f1", "City A", 10.0, 10.0, priority=100.0, width=15.0, height=8.0),
            PointFeature("f2", "City B", 12.0, 11.0, priority=80.0, width=15.0, height=8.0),
            PointFeature("f3", "Town C", 50.0, 50.0, priority=50.0, width=15.0, height=8.0),
        ]
        res = solve_label_placement(features)
        self.assertGreater(res.total_labels_placed, 0)
        self.assertEqual(res.conflict_count, 0)
        self.assertGreater(res.aesthetic_score, 0.0)

    def test_p2_capacitated_p_median(self):
        demands = [
            DemandPoint("d1", 0.0, 0.0, 10.0),
            DemandPoint("d2", 2.0, 1.0, 15.0),
            DemandPoint("d3", 20.0, 20.0, 12.0),
            DemandPoint("d4", 22.0, 19.0, 18.0),
        ]
        candidates = [
            CandidateFacility("fac_1", 1.0, 1.0, capacity=50.0),
            CandidateFacility("fac_2", 21.0, 20.0, capacity=50.0),
            CandidateFacility("fac_3", 100.0, 100.0, capacity=50.0),
        ]
        res = solve_capacitated_p_median(demands, candidates, p=2)
        self.assertEqual(len(res.open_facilities), 2)
        self.assertEqual(len(res.assignments), 4)
        self.assertGreater(res.capacity_utilization_pct, 0.0)

    def test_p3_terrain_viewshed_siting(self):
        # 5x5 elevation grid with hill at (2, 2)
        grid = [TerrainPoint(x, y, 10.0 if (x == 2 and y == 2) else 1.0)
                for x in range(5) for y in range(5)]
        radars = [(0, 0), (2, 2), (4, 4)]
        res = solve_viewshed_siting(grid, radars, sensor_range=5.0, k_sensors=2)
        self.assertGreater(len(res.selected_sensors), 0)
        self.assertGreater(res.total_covered_cells, 0)
        self.assertGreater(res.coverage_percentage, 0.0)

    def test_p4_territorial_districting(self):
        # 6 linearly adjacent tracts
        tracts = [
            SpatialTract("t0", 0.0, 0.0, 100.0, ["t1"]),
            SpatialTract("t1", 1.0, 0.0, 105.0, ["t0", "t2"]),
            SpatialTract("t2", 2.0, 0.0, 95.0, ["t1", "t3"]),
            SpatialTract("t3", 3.0, 0.0, 100.0, ["t2", "t4"]),
            SpatialTract("t4", 4.0, 0.0, 110.0, ["t3", "t5"]),
            SpatialTract("t5", 5.0, 0.0, 90.0, ["t4"]),
        ]
        res = solve_territorial_districting(tracts, k_districts=2)
        self.assertEqual(len(res.districts), 2)
        self.assertTrue(res.contiguity_certified)
        self.assertGreater(res.compactness_score, 0.0)

    def test_p5_dubins_uav_path(self):
        wps = [
            UAVWaypoint("w1", 0.0, 0.0, 0.0),
            UAVWaypoint("w2", 10.0, 0.0, 45.0),
            UAVWaypoint("w3", 10.0, 10.0, 90.0),
            UAVWaypoint("w4", 0.0, 10.0, 180.0),
        ]
        threats = [ThreatDome(5.0, 5.0, radius=3.0, lethality=2.0)]
        res = solve_dubins_uav_path(wps, threats, min_turn_radius=2.0)
        self.assertEqual(len(res.waypoint_order), 4)
        self.assertTrue(res.kinematic_feasible)
        self.assertGreater(res.total_path_length, 0.0)

    def test_p6_line_simplification(self):
        # Line with 10 collinear points plus small jitter
        pts = [(float(i), 0.1 if i % 2 == 1 else 0.0) for i in range(10)]
        res = solve_line_simplification(pts, epsilon=0.5)
        self.assertLess(len(res.simplified_points), len(pts))
        self.assertEqual(res.simplified_points[0], pts[0])
        self.assertEqual(res.simplified_points[-1], pts[-1])
        self.assertTrue(res.self_intersection_free)

    def test_p7_evacuation_routing(self):
        zones = [
            EvacuationZone("z1", 200, 2.0),
            EvacuationZone("z2", 300, 3.0),
        ]
        roads = [
            RoadSegment("z1", "junc", travel_time_base_min=5.0, capacity_flow=500.0),
            RoadSegment("z2", "junc", travel_time_base_min=4.0, capacity_flow=500.0),
            RoadSegment("junc", "haven", travel_time_base_min=10.0, capacity_flow=800.0),
        ]
        res = solve_evacuation_routing(zones, roads, ["haven"])
        self.assertIn("z1", res.evacuation_paths)
        self.assertEqual(res.evacuation_paths["z1"][-1], "haven")
        self.assertGreater(res.clearance_time_min, 0.0)

    def test_p8_spatial_weights_optimization(self):
        features = [
            SpatialFeature(f"pt_{i}", float(i), float(i), val=10.0 * i)
            for i in range(10)
        ]
        res = solve_spatial_weights_optimization(features, candidate_k_values=[2, 3])
        self.assertIsInstance(res.morans_i, float)
        self.assertIsInstance(res.z_score, float)
        self.assertGreaterEqual(res.matrix_sparsity_pct, 0.0)

    def test_p9_spatial_colocation_mining(self):
        # 3 co-located pairs
        instances = [
            SpatialInstance("a1", "Hospital", 1.0, 1.0),
            SpatialInstance("b1", "Pharmacy", 1.5, 1.2),
            SpatialInstance("a2", "Hospital", 10.0, 10.0),
            SpatialInstance("b2", "Pharmacy", 10.2, 10.3),
        ]
        res = solve_colocation_mining(instances, distance_threshold=2.0, min_prevalence=0.5)
        self.assertGreaterEqual(res.total_patterns_found, 1)

    def test_p10_satellite_tasking(self):
        targets = [
            ObservationTarget("t1", 0.0, 0.0, priority=50.0, window_start=10.0, window_end=50.0, duration=10.0),
            ObservationTarget("t2", 5.0, 5.0, priority=80.0, window_start=25.0, window_end=70.0, duration=10.0),
            ObservationTarget("t3", 100.0, 100.0, priority=30.0, window_start=15.0, window_end=90.0, duration=10.0),
        ]
        res = solve_satellite_tasking(targets, slew_rate_deg_per_sec=2.0)
        self.assertGreater(len(res.scheduled_tasks), 0)
        self.assertTrue(res.slew_transition_certified)

    def test_engine_full_benchmark(self):
        engine = GeospatialNPHardEngine()
        bench = engine.benchmark_all_10()
        self.assertEqual(len(bench["solvers"]), 10)
        self.assertGreater(bench["total_benchmark_time_us"], 0.0)

    def test_defense_and_urban_adapters(self):
        engine = GeospatialNPHardEngine()
        suite = engine.generate_synthetic_benchmark_suite()
        # Test defense adapter
        uav_res = engine.defense_adapter.plan_stealth_recon_flight(*suite["dtspn"])
        self.assertTrue(uav_res.kinematic_feasible)
        # Test urban adapter
        palp_res = engine.urban_adapter.render_cartographic_labels(suite["palp"])
        self.assertGreater(palp_res.total_labels_placed, 0)


if __name__ == "__main__":
    unittest.main()
