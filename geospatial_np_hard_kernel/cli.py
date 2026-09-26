"""Command Line Interface for Geospatial NP-Hard Kernel."""
import argparse
import json
import sys
from .engine import GeospatialNPHardEngine


def run_benchmark_all():
    engine = GeospatialNPHardEngine()
    print("=" * 80)
    print("  GEOSPATIAL NP-HARD KERNEL: 10 APEX GIS & GEOINT SOLVERS BENCHMARK")
    print("=" * 80)
    
    results = engine.benchmark_all_10()
    solvers = results["solvers"]

    print(f"\n{'Solver ID & Name':<35} | {'Key Metric':<25} | {'Latency':<12}")
    print("-" * 80)
    for s_name, data in solvers.items():
        time_str = f"{data['time_us']:.1f} µs"
        # Extract primary metric
        primary_metric = [f"{k}={v}" for k, v in data.items() if k != 'time_us'][0]
        print(f"{s_name:<35} | {primary_metric:<25} | {time_str:<12}")

    print("-" * 80)
    print(f"Total Combined Pipeline Benchmark Execution: {results['total_benchmark_time_us']:.1f} µs")
    print("=" * 80)
    return results


def run_defense_demo():
    engine = GeospatialNPHardEngine()
    print("=" * 80)
    print("  DEFENSE GEOSPATIAL INTELLIGENCE (GEOINT) & SURVEILLANCE HARNESS")
    print("=" * 80)
    suite = engine.generate_synthetic_benchmark_suite()
    
    # UAV Path
    res_uav = engine.defense_adapter.plan_stealth_recon_flight(*suite["dtspn"])
    print(f"[1] Stealth Recon UAV Path: {len(res_uav.waypoint_order)} wps, "
          f"length={res_uav.total_path_length}, threat_penalty={res_uav.threat_exposure_penalty} ({res_uav.execution_time_us} µs)")

    # Radar Viewshed
    res_radar = engine.defense_adapter.deploy_border_radar_network(*suite["viewshed"])
    print(f"[2] Border Radar Viewshed: {len(res_radar.selected_sensors)} radars, "
          f"coverage={res_radar.coverage_percentage}%, cells={res_radar.total_covered_cells} ({res_radar.execution_time_us} µs)")

    # Satellite Tasking
    res_sat = engine.defense_adapter.task_recon_constellation(*suite["satellite"])
    print(f"[3] Recon Satellite Constellation: {len(res_sat.scheduled_tasks)} tasks, "
          f"priority={res_sat.total_collected_priority}, slew_certified={res_sat.slew_transition_certified} ({res_sat.execution_time_us} µs)")

    # SIGINT Colocation
    res_sigint = engine.defense_adapter.mine_adversarial_signatures(*suite["colocation"])
    print(f"[4] SIGINT / ELINT Signature Colocation: {res_sigint.total_patterns_found} patterns found ({res_sigint.execution_time_us} µs)")
    print("=" * 80)


def run_urban_demo():
    engine = GeospatialNPHardEngine()
    print("=" * 80)
    print("  URBAN CIVILIAN GIS & INFRASTRUCTURE PLANNING (ARCGIS / QGIS) HARNESS")
    print("=" * 80)
    suite = engine.generate_synthetic_benchmark_suite()

    # PALP Labels
    res_palp = engine.urban_adapter.render_cartographic_labels(suite["palp"])
    print(f"[1] Cadastral Label Placement: {res_palp.total_labels_placed} labels, "
          f"aesthetic_score={res_palp.aesthetic_score}/100 ({res_palp.execution_time_us} µs)")

    # EMS Facilities
    res_ems = engine.urban_adapter.plan_emergency_medical_stations(*suite["cpmp"])
    print(f"[2] Emergency EMS Stations: {len(res_ems.open_facilities)} opened, "
          f"utilization={res_ems.capacity_utilization_pct}%, cost={res_ems.total_weighted_distance} ({res_ems.execution_time_us} µs)")

    # Wards Districting
    res_dist = engine.urban_adapter.partition_municipal_wards(*suite["districting"])
    print(f"[3] Municipal Ward Districting: {len(res_dist.districts)} wards, "
          f"pop_disparity={res_dist.population_disparity_pct}%, contiguous={res_dist.contiguity_certified} ({res_dist.execution_time_us} µs)")

    # Evacuation Routing
    res_evac = engine.urban_adapter.schedule_civil_evacuation(*suite["evacuation"])
    print(f"[4] Hurricane Evacuation Egress: clearance={res_evac.clearance_time_min} min, "
          f"bottleneck_ratio={res_evac.chokepoint_bottleneck_ratio} ({res_evac.execution_time_us} µs)")

    # Line Simplification
    res_simp = engine.urban_adapter.generalize_shorelines_or_cadastre(*suite["simplification"])
    print(f"[5] Boundary Polyline Generalization: vertex_reduction={res_simp.vertex_reduction_pct}%, "
          f"frechet_dev={res_simp.max_frechet_deviation} ({res_simp.execution_time_us} µs)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Geospatial NP-Hard Kernel CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("benchmark-all", help="Benchmark all 10 NP-hard GIS/GEOINT solvers")
    subparsers.add_parser("defense-demo", help="Run Defense GEOINT & Multi-Domain Surveillance demo")
    subparsers.add_parser("urban-demo", help="Run Urban Civilian ArcGIS / Municipal Planning demo")

    args = parser.parse_args()
    if args.command == "benchmark-all" or args.command is None:
        run_benchmark_all()
    elif args.command == "defense-demo":
        run_defense_demo()
    elif args.command == "urban-demo":
        run_urban_demo()


if __name__ == "__main__":
    main()
