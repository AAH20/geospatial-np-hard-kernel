"""Spatial Contiguity-Constrained Territorial Districting Solver.

Solves the NP-hard spatial districting problem: partitions spatial census tracts/zones into
k contiguous, population-balanced, and compact districts.
Uses seed-growing region expansion with boundary graph swap optimization and BFS contiguity certification.
"""
import math
import time
from collections import deque
from typing import List, Dict, Set, Tuple
from .models import SpatialTract, TerritorialDistrictingResult


def _is_connected(tract_ids: Set[str], adj: Dict[str, Set[str]]) -> bool:
    """Verifies that the induced subgraph of tract_ids is connected via BFS."""
    if not tract_ids:
        return True
    start = next(iter(tract_ids))
    visited = {start}
    q = deque([start])
    while q:
        curr = q.popleft()
        for neighbor in adj.get(curr, set()):
            if neighbor in tract_ids and neighbor not in visited:
                visited.add(neighbor)
                q.append(neighbor)
    return len(visited) == len(tract_ids)


def solve_territorial_districting(
    tracts: List[SpatialTract],
    k_districts: int = 3
) -> TerritorialDistrictingResult:
    """Solves contiguity-constrained territorial districting via seeded region growing + boundary exchange."""
    t0 = time.perf_counter()
    if not tracts or k_districts <= 0:
        return TerritorialDistrictingResult(
            districts={},
            population_disparity_pct=0.0,
            contiguity_certified=True,
            compactness_score=100.0,
            algorithm="Contiguity-Preserving-RegionGrow",
            execution_time_us=0.0
        )

    k = min(k_districts, len(tracts))
    tract_map = {t.tract_id: t for t in tracts}
    adj: Dict[str, Set[str]] = {t.tract_id: set(t.neighbors) for t in tracts}
    total_pop = sum(t.population for t in tracts)
    target_pop = total_pop / float(k)

    # 1. Farthest-point sampling to choose k seeds
    seeds: List[str] = [tracts[0].tract_id]
    while len(seeds) < k:
        best_cand = None
        max_min_dist = -1.0
        for t in tracts:
            if t.tract_id in seeds:
                continue
            min_dist = min(math.hypot(t.x - tract_map[s].x, t.y - tract_map[s].y) for s in seeds)
            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_cand = t.tract_id
        if best_cand:
            seeds.append(best_cand)
        else:
            break

    # Initialize districts
    district_assignments: Dict[str, int] = {}
    districts: Dict[int, Set[str]] = {i: {seed} for i, seed in enumerate(seeds)}
    district_pops: Dict[int, float] = {i: tract_map[seed].population for i, seed in enumerate(seeds)}
    for i, seed in enumerate(seeds):
        district_assignments[seed] = i

    unassigned = set(tract_map.keys()) - set(seeds)

    # 2. Region growing: iteratively expand lowest population district with adjacent unassigned tract
    while unassigned:
        # Sort districts by current population ascending
        sorted_dists = sorted(districts.keys(), key=lambda d_id: district_pops[d_id])
        assigned_one = False
        
        for d_id in sorted_dists:
            # Find candidate adjacent unassigned tracts
            cand_tracts = set()
            for t_id in districts[d_id]:
                for n_id in adj.get(t_id, set()):
                    if n_id in unassigned:
                        cand_tracts.add(n_id)
            
            if cand_tracts:
                # Pick candidate closest to district centroid
                d_tracts = [tract_map[tid] for tid in districts[d_id]]
                cx = sum(t.x for t in d_tracts) / len(d_tracts)
                cy = sum(t.y for t in d_tracts) / len(d_tracts)
                best_cand = min(cand_tracts, key=lambda tid: math.hypot(tract_map[tid].x - cx, tract_map[tid].y - cy))
                
                districts[d_id].add(best_cand)
                district_pops[d_id] += tract_map[best_cand].population
                district_assignments[best_cand] = d_id
                unassigned.remove(best_cand)
                assigned_one = True
                break

        if not assigned_one and unassigned:
            # Handle disconnected disconnected component: assign to closest district
            orphan = unassigned.pop()
            closest_d = min(districts.keys(), key=lambda d_id: min(
                math.hypot(tract_map[orphan].x - tract_map[tid].x, tract_map[orphan].y - tract_map[tid].y)
                for tid in districts[d_id]
            ))
            districts[closest_d].add(orphan)
            district_pops[closest_d] += tract_map[orphan].population
            district_assignments[orphan] = closest_d

    # 3. Boundary local search: swap tracts to balance population while preserving contiguity
    for _ in range(5):
        for t_id, d_id in list(district_assignments.items()):
            # Check if t_id can move to neighboring district
            for n_id in adj.get(t_id, set()):
                target_d = district_assignments[n_id]
                if target_d != d_id:
                    # Check if donor district remains connected without t_id
                    trial_donor = districts[d_id] - {t_id}
                    if trial_donor and _is_connected(trial_donor, adj):
                        # Check population disparity delta
                        curr_disp = abs(district_pops[d_id] - target_pop) + abs(district_pops[target_d] - target_pop)
                        new_disp = abs((district_pops[d_id] - tract_map[t_id].population) - target_pop) + \
                                   abs((district_pops[target_d] + tract_map[t_id].population) - target_pop)
                        if new_disp < curr_disp - 1e-4:
                            districts[d_id].remove(t_id)
                            districts[target_d].add(t_id)
                            district_pops[d_id] -= tract_map[t_id].population
                            district_pops[target_d] += tract_map[t_id].population
                            district_assignments[t_id] = target_d
                            break

    # 4. Certify contiguity and calculate compactness
    all_connected = all(_is_connected(d_set, adj) for d_set in districts.values())
    max_disparity = max(abs(pop - target_pop) for pop in district_pops.values()) if target_pop > 0 else 0.0
    disparity_pct = (max_disparity / target_pop) * 100.0 if target_pop > 0 else 0.0

    # Compactness: inverse of average inertia (distance to district centroid)
    total_dispersion = 0.0
    for d_id, d_set in districts.items():
        if not d_set:
            continue
        cx = sum(tract_map[tid].x for tid in d_set) / len(d_set)
        cy = sum(tract_map[tid].y for tid in d_set) / len(d_set)
        total_dispersion += sum(math.hypot(tract_map[tid].x - cx, tract_map[tid].y - cy) for tid in d_set)
    compactness = 100.0 / (1.0 + total_dispersion / max(1, len(tracts)))

    t1 = time.perf_counter()
    exec_us = (t1 - t0) * 1_000_000.0

    return TerritorialDistrictingResult(
        districts={d_id: sorted(list(t_ids)) for d_id, t_ids in districts.items()},
        population_disparity_pct=round(disparity_pct, 2),
        contiguity_certified=all_connected,
        compactness_score=round(compactness, 2),
        algorithm="Contiguity-Preserving-RegionGrow",
        execution_time_us=round(exec_us, 2)
    )
