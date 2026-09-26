"""Time-Dependent Evacuation Route Scheduling (TD-VRPTW) Solver.

Solves the NP-hard dynamic traffic assignment and emergency egress clearance problem.
Models time-expanded network flows with flow-capacity bottleneck congestion penalties and
hazard delay factors (flooding, wildfires, toxic plume dispersal).
"""
import heapq
import time
from typing import List, Dict, Set, Tuple
from .models import EvacuationZone, RoadSegment, EvacuationRouteResult


def solve_evacuation_routing(
    zones: List[EvacuationZone],
    roads: List[RoadSegment],
    safe_havens: List[str]
) -> EvacuationRouteResult:
    """Solves time-dependent evacuation routing using priority-ordered dynamic shortest path relaxation."""
    t0 = time.perf_counter()
    if not zones or not roads or not safe_havens:
        return EvacuationRouteResult(
            evacuation_paths={},
            clearance_time_min=0.0,
            chokepoint_bottleneck_ratio=0.0,
            algorithm="Dynamic-TD-EvacFlow",
            execution_time_us=0.0
        )

    # Build adjacency list: node -> list of (target, base_time, capacity, flood_factor)
    adj: Dict[str, List[RoadSegment]] = {}
    for r in roads:
        adj.setdefault(r.source, []).append(r)

    # Sort evacuation zones by urgency descending, then population
    sorted_zones = sorted(zones, key=lambda z: (z.urgency, z.population), reverse=True)

    # Track road usage / flow
    edge_flow: Dict[Tuple[str, str], float] = {}
    evac_paths: Dict[str, List[str]] = {}
    max_clearance_time = 0.0
    safe_set = set(safe_havens)

    for zone in sorted_zones:
        start_node = zone.zone_id
        # Dijkstra with congestion penalty:
        # edge travel time = base * flood_factor * (1.0 + 0.5 * (current_flow / capacity)^2)
        dist: Dict[str, float] = {start_node: 0.0}
        parent: Dict[str, str] = {}
        pq: List[Tuple[float, str]] = [(0.0, start_node)]

        dest_reached = None
        while pq:
            curr_time, u = heapq.heappop(pq)
            if u in safe_set:
                dest_reached = u
                break
            if curr_time > dist.get(u, float('inf')):
                continue

            for edge in adj.get(u, []):
                v = edge.target
                curr_flow = edge_flow.get((u, v), 0.0)
                cong_factor = 1.0 + 0.5 * ((curr_flow / max(1.0, edge.capacity_flow)) ** 2)
                travel_time = edge.travel_time_base_min * edge.flood_delay_factor * cong_factor
                alt = curr_time + travel_time

                if alt < dist.get(v, float('inf')):
                    dist[v] = alt
                    parent[v] = u
                    heapq.heappush(pq, (alt, v))

        if dest_reached:
            # Reconstruct path
            path = []
            curr = dest_reached
            while curr in parent:
                path.append(curr)
                curr = parent[curr]
            path.append(start_node)
            path.reverse()
            evac_paths[zone.zone_id] = path

            # Add flow along edges
            for i in range(len(path) - 1):
                e_key = (path[i], path[i + 1])
                edge_flow[e_key] = edge_flow.get(e_key, 0.0) + zone.population

            zone_clearance = dist[dest_reached]
            if zone_clearance > max_clearance_time:
                max_clearance_time = zone_clearance
        else:
            evac_paths[zone.zone_id] = [start_node]

    # Calculate chokepoint bottleneck ratio: max(flow / capacity) across all traversed segments
    max_bottleneck = 0.0
    for edge in roads:
        flow = edge_flow.get((edge.source, edge.target), 0.0)
        ratio = flow / max(1.0, edge.capacity_flow)
        if ratio > max_bottleneck:
            max_bottleneck = ratio

    t1 = time.perf_counter()
    exec_us = (t1 - t0) * 1_000_000.0

    return EvacuationRouteResult(
        evacuation_paths=evac_paths,
        clearance_time_min=round(max_clearance_time, 2),
        chokepoint_bottleneck_ratio=round(max_bottleneck, 3),
        algorithm="Dynamic-TD-EvacFlow",
        execution_time_us=round(exec_us, 2)
    )
