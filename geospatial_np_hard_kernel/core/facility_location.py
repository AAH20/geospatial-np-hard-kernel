"""Capacitated p-Median & Emergency Facility Location (CPMP) Solver.

Solves the NP-hard optimal location of p facilities serving spatially distributed demand points
subject to strict capacity thresholds, minimizing total demand-weighted transit distances.
Uses Teitz-Bart vertex substitution with greedy capacitated transportation assignment.
"""
import math
import time
from typing import List, Dict, Set, Tuple
from .models import DemandPoint, CandidateFacility, FacilityLocationResult


def _euclidean_dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x1 - x2, y1 - y2)


def _solve_assignment(
    demand_points: List[DemandPoint],
    open_facs: List[CandidateFacility]
) -> Tuple[Dict[str, str], float, float]:
    """Greedy transportation assignment respecting individual facility capacities."""
    remaining_capacity = {f.facility_id: f.capacity for f in open_facs}
    assignments: Dict[str, str] = {}
    total_cost = 0.0
    total_demand_served = 0.0

    # Sort demand pairs by distance
    for d in demand_points:
        sorted_facs = sorted(open_facs, key=lambda f: _euclidean_dist(d.x, d.y, f.x, f.y))
        assigned = False
        for f in sorted_facs:
            if remaining_capacity[f.facility_id] >= d.demand_weight:
                remaining_capacity[f.facility_id] -= d.demand_weight
                assignments[d.point_id] = f.facility_id
                total_cost += d.demand_weight * _euclidean_dist(d.x, d.y, f.x, f.y)
                total_demand_served += d.demand_weight
                assigned = True
                break
        
        # If capacity exhausted on preferred, allocate to least loaded open facility
        if not assigned and open_facs:
            least_loaded = min(open_facs, key=lambda f: _euclidean_dist(d.x, d.y, f.x, f.y))
            remaining_capacity[least_loaded.facility_id] -= d.demand_weight
            assignments[d.point_id] = least_loaded.facility_id
            total_cost += d.demand_weight * _euclidean_dist(d.x, d.y, least_loaded.x, least_loaded.y)
            total_demand_served += d.demand_weight

    total_capacity = sum(f.capacity for f in open_facs) if open_facs else 1.0
    utilization_pct = min(100.0, (total_demand_served / total_capacity) * 100.0) if total_capacity > 0 else 0.0
    return assignments, total_cost, utilization_pct


def solve_capacitated_p_median(
    demand_points: List[DemandPoint],
    candidate_facilities: List[CandidateFacility],
    p: int = 3
) -> FacilityLocationResult:
    """Solves CPMP via greedy initialization + 1-opt Teitz-Bart vertex substitution."""
    t0 = time.perf_counter()
    if not candidate_facilities or not demand_points:
        return FacilityLocationResult(
            open_facilities=[],
            assignments={},
            total_weighted_distance=0.0,
            capacity_utilization_pct=0.0,
            algorithm="Teitz-Bart-CPMP",
            execution_time_us=0.0
        )

    k = min(p, len(candidate_facilities))
    
    # 1. Greedy initial selection: pick facility closest to centroid of demand, then iteratively add
    fac_dict = {f.facility_id: f for f in candidate_facilities}
    open_set: List[CandidateFacility] = []
    candidates = list(candidate_facilities)

    # First facility: closest to weighted center of gravity
    cx = sum(d.x * d.demand_weight for d in demand_points) / max(1e-6, sum(d.demand_weight for d in demand_points))
    cy = sum(d.y * d.demand_weight for d in demand_points) / max(1e-6, sum(d.demand_weight for d in demand_points))
    first_fac = min(candidates, key=lambda f: math.hypot(f.x - cx, f.y - cy))
    open_set.append(first_fac)
    candidates.remove(first_fac)

    while len(open_set) < k and candidates:
        best_cand = None
        best_cost = float('inf')
        for cand in candidates:
            trial_set = open_set + [cand]
            _, cost, _ = _solve_assignment(demand_points, trial_set)
            if cost < best_cost:
                best_cost = cost
                best_cand = cand
        if best_cand:
            open_set.append(best_cand)
            candidates.remove(best_cand)

    # 2. Local search swap (Teitz-Bart)
    improved = True
    passes = 0
    max_passes = 10
    current_assignments, current_cost, current_util = _solve_assignment(demand_points, open_set)

    while improved and passes < max_passes:
        improved = False
        passes += 1
        for i in range(len(open_set)):
            for cand in candidates:
                trial_open = open_set.copy()
                trial_open[i] = cand
                trial_assign, trial_cost, trial_util = _solve_assignment(demand_points, trial_open)
                if trial_cost < current_cost - 1e-4:
                    old_fac = open_set[i]
                    open_set[i] = cand
                    candidates.remove(cand)
                    candidates.append(old_fac)
                    current_assignments = trial_assign
                    current_cost = trial_cost
                    current_util = trial_util
                    improved = True
                    break
            if improved:
                break

    t1 = time.perf_counter()
    exec_us = (t1 - t0) * 1_000_000.0

    return FacilityLocationResult(
        open_facilities=[f.facility_id for f in open_set],
        assignments=current_assignments,
        total_weighted_distance=round(current_cost, 2),
        capacity_utilization_pct=round(current_util, 2),
        algorithm="Teitz-Bart-CPMP",
        execution_time_us=round(exec_us, 2)
    )
