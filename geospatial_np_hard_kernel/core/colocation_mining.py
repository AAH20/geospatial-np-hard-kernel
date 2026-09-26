"""Spatial Co-Location Pattern Mining Solver.

Discovers non-trivial boolean spatial association rules and co-location patterns
using Apriori-like candidate generation with participation index (PI) monotonic pruning
and geometric clique verification.
"""
import itertools
import math
import time
from typing import List, Dict, Set, Tuple
from .models import SpatialInstance, CoLocationPatternResult


def _distance(i1: SpatialInstance, i2: SpatialInstance) -> float:
    return math.hypot(i1.x - i2.x, i1.y - i2.y)


def solve_colocation_mining(
    instances: List[SpatialInstance],
    distance_threshold: float = 10.0,
    min_prevalence: float = 0.5
) -> CoLocationPatternResult:
    """Discovers prevalent spatial co-location patterns with participation index >= min_prevalence."""
    t0 = time.perf_counter()
    if not instances:
        return CoLocationPatternResult(
            prevalent_patterns=[],
            total_patterns_found=0,
            min_prevalence_threshold=min_prevalence,
            algorithm="Apriori-Spatial-Clique-Mining",
            execution_time_us=0.0
        )

    # Group instances by feature type
    by_type: Dict[str, List[SpatialInstance]] = {}
    for inst in instances:
        by_type.setdefault(inst.feature_type, []).append(inst)

    types = sorted(list(by_type.keys()))
    if len(types) < 2:
        return CoLocationPatternResult(
            prevalent_patterns=[],
            total_patterns_found=0,
            min_prevalence_threshold=min_prevalence,
            algorithm="Apriori-Spatial-Clique-Mining",
            execution_time_us=0.0
        )

    # Precompute pairwise neighbor lookup between instances of different types
    type_instance_map = {inst.instance_id: inst for inst in instances}
    neighbors: Dict[str, Set[str]] = {inst.instance_id: set() for inst in instances}
    
    for i in range(len(instances)):
        for j in range(i + 1, len(instances)):
            i1, i2 = instances[i], instances[j]
            if i1.feature_type != i2.feature_type:
                if _distance(i1, i2) <= distance_threshold:
                    neighbors[i1.instance_id].add(i2.instance_id)
                    neighbors[i2.instance_id].add(i1.instance_id)

    prevalent_patterns: List[Tuple[Tuple[str, ...], float]] = []

    # Level 2 candidate generation: pairs of types
    curr_patterns: List[Tuple[str, ...]] = []
    for c2 in itertools.combinations(types, 2):
        t1, t2 = c2
        insts1 = by_type[t1]
        insts2 = by_type[t2]
        
        part1 = sum(1 for i1 in insts1 if any(i2.instance_id in neighbors[i1.instance_id] for i2 in insts2))
        part2 = sum(1 for i2 in insts2 if any(i1.instance_id in neighbors[i2.instance_id] for i1 in insts1))
        
        pr1 = part1 / float(len(insts1)) if insts1 else 0.0
        pr2 = part2 / float(len(insts2)) if insts2 else 0.0
        pi = min(pr1, pr2)

        if pi >= min_prevalence:
            prevalent_patterns.append((c2, round(pi, 3)))
            curr_patterns.append(c2)

    # Level 3 candidate generation
    if len(curr_patterns) >= 3:
        # Candidate 3-patterns from joining prevalent 2-patterns
        c3_cands = set()
        for p1, p2 in itertools.combinations(curr_patterns, 2):
            merged = tuple(sorted(list(set(p1 + p2))))
            if len(merged) == 3:
                # Check if all 2-subsets are prevalent
                sub_pairs = list(itertools.combinations(merged, 2))
                if all(sp in curr_patterns for sp in sub_pairs):
                    c3_cands.add(merged)

        for c3 in c3_cands:
            t1, t2, t3 = c3
            participating: Dict[str, Set[str]] = {t: set() for t in c3}
            # Search for triangles
            for i1 in by_type[t1]:
                n1 = neighbors[i1.instance_id]
                for i2 in by_type[t2]:
                    if i2.instance_id in n1:
                        n2 = neighbors[i2.instance_id]
                        for i3 in by_type[t3]:
                            if i3.instance_id in n1 and i3.instance_id in n2:
                                participating[t1].add(i1.instance_id)
                                participating[t2].add(i2.instance_id)
                                participating[t3].add(i3.instance_id)
            
            pr_vals = [len(participating[t]) / float(len(by_type[t])) for t in c3]
            pi = min(pr_vals) if pr_vals else 0.0
            if pi >= min_prevalence:
                prevalent_patterns.append((c3, round(pi, 3)))

    t1 = time.perf_counter()
    exec_us = (t1 - t0) * 1_000_000.0

    return CoLocationPatternResult(
        prevalent_patterns=prevalent_patterns,
        total_patterns_found=len(prevalent_patterns),
        min_prevalence_threshold=min_prevalence,
        algorithm="Apriori-Spatial-Clique-Mining",
        execution_time_us=round(exec_us, 2)
    )
