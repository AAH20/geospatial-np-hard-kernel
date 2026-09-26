"""Art Gallery & Terrain Viewshed Radar Siting Solver.

Solves the NP-hard terrain viewshed sensor placement problem using ray-casted line-of-sight (LOS)
analysis coupled with submodular greedy set-coverage optimization with (1 - 1/e) optimality guarantee.
"""
import math
import time
from typing import List, Dict, Set, Tuple, Optional
from .models import TerrainPoint, ViewshedResult


def _compute_cell_los(
    ox: int, oy: int, oz: float,
    tx: int, ty: int, tz: float,
    elevation_map: Dict[Tuple[int, int], float]
) -> bool:
    """Ray casts between observer (ox, oy, oz) and target (tx, ty, tz) to test LOS obstruction."""
    dx = tx - ox
    dy = ty - oy
    dist = math.hypot(dx, dy)
    if dist <= 1.0:
        return True

    steps = max(abs(dx), abs(dy))
    x_inc = dx / float(steps)
    y_inc = dy / float(steps)

    cx = float(ox)
    cy = float(oy)

    for i in range(1, steps):
        cx += x_inc
        cy += y_inc
        ix, iy = int(round(cx)), int(round(cy))
        curr_elev = elevation_map.get((ix, iy), 0.0)
        
        # Line of sight ray elevation at current fraction
        t = i / float(steps)
        ray_elev = oz + t * (tz - oz)
        if curr_elev > ray_elev:
            return False  # Obstructed
    return True


def solve_viewshed_siting(
    grid_points: List[TerrainPoint],
    candidate_sensors: List[Tuple[int, int]],
    sensor_range: float = 10.0,
    k_sensors: int = 3,
    sensor_mast_height: float = 2.0
) -> ViewshedResult:
    """Solves the terrain viewshed coverage problem via submodular greedy approximation."""
    t0 = time.perf_counter()
    if not grid_points or not candidate_sensors:
        return ViewshedResult(
            selected_sensors=[],
            total_covered_cells=0,
            coverage_percentage=0.0,
            redundancy_overlap_ratio=0.0,
            algorithm="Submodular-Greedy-Viewshed",
            execution_time_us=0.0
        )

    # Build elevation lookup
    elev_map = {(p.x, p.y): p.elevation for p in grid_points}
    all_cells = set(elev_map.keys())

    # Precompute coverage sets for each candidate sensor
    sensor_viewsheds: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {}
    for (sx, sy) in candidate_sensors:
        s_elev = elev_map.get((sx, sy), 0.0) + sensor_mast_height
        covered: Set[Tuple[int, int]] = set()
        for (tx, ty), t_elev in elev_map.items():
            if math.hypot(tx - sx, ty - sy) <= sensor_range:
                if _compute_cell_los(sx, sy, s_elev, tx, ty, t_elev, elev_map):
                    covered.add((tx, ty))
        sensor_viewsheds[(sx, sy)] = covered

    # Submodular greedy selection
    selected: List[Tuple[int, int]] = []
    uncovered = set(all_cells)
    total_coverage_events = 0
    k = min(k_sensors, len(candidate_sensors))

    for _ in range(k):
        best_cand = None
        best_gain = -1
        for cand, cov in sensor_viewsheds.items():
            if cand in selected:
                continue
            gain = len(cov.intersection(uncovered))
            if gain > best_gain:
                best_gain = gain
                best_cand = cand
        if best_cand and best_gain > 0:
            selected.append(best_cand)
            cov_set = sensor_viewsheds[best_cand]
            total_coverage_events += len(cov_set)
            uncovered.difference_update(cov_set)
        elif best_cand:
            # Add even if marginal gain is zero to fulfill k
            selected.append(best_cand)

    covered_cells_count = len(all_cells) - len(uncovered)
    coverage_pct = (covered_cells_count / len(all_cells)) * 100.0 if all_cells else 0.0
    redundancy_ratio = (total_coverage_events / max(1, covered_cells_count)) if covered_cells_count > 0 else 0.0

    t1 = time.perf_counter()
    exec_us = (t1 - t0) * 1_000_000.0

    return ViewshedResult(
        selected_sensors=selected,
        total_covered_cells=covered_cells_count,
        coverage_percentage=round(coverage_pct, 2),
        redundancy_overlap_ratio=round(redundancy_ratio, 2),
        algorithm="Submodular-Greedy-Viewshed",
        execution_time_us=round(exec_us, 2)
    )
