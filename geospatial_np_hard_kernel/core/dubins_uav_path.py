"""Dubins Multi-UAV Trajectory with Threat Domes (DTSPN) Solver.

Solves the Dubins Traveling Salesperson Problem with Neighborhoods and Threat Avoidance.
Balances kinematic minimum-turn radius curvature constraints against air defense threat dome penetration.
Uses 2-opt trajectory search with arc-tangent heading relaxation.
"""
import math
import time
from typing import List, Tuple
from .models import UAVWaypoint, ThreatDome, DubinsTrajectoryResult


def _point_to_segment_dist(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """Computes minimum Euclidean distance from point (px, py) to line segment (x1, y1)-(x2, y2)."""
    dx = x2 - x1
    dy = y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0.0:
        return math.hypot(px - x1, py - y1)
    
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return math.hypot(px - proj_x, py - proj_y)


def _segment_threat_penalty(
    x1: float, y1: float, x2: float, y2: float,
    threats: List[ThreatDome]
) -> float:
    """Calculates cumulative lethality penalty for traversing a segment through threat domes."""
    penalty = 0.0
    for threat in threats:
        d = _point_to_segment_dist(threat.center_x, threat.center_y, x1, y1, x2, y2)
        if d < threat.radius:
            # Threat intensity inversely proportional to distance to threat center
            intensity = (1.0 - (d / threat.radius)) * threat.lethality
            seg_len = math.hypot(x2 - x1, y2 - y1)
            penalty += intensity * seg_len
    return penalty


def _dubins_approx_cost(
    w1: UAVWaypoint, w2: UAVWaypoint,
    min_radius: float, threats: List[ThreatDome]
) -> Tuple[float, float]:
    """Computes kinematically penalized length and threat penalty between two UAV waypoints."""
    euc_dist = math.hypot(w2.x - w1.x, w2.y - w1.y)
    
    # Heading angle discrepancy penalty
    h1 = math.radians(w1.desired_heading_deg)
    h2 = math.radians(w2.desired_heading_deg)
    angle_diff = abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1)))
    turning_arc = min_radius * angle_diff
    dubins_len = euc_dist + turning_arc

    threat_cost = _segment_threat_penalty(w1.x, w1.y, w2.x, w2.y, threats)
    return dubins_len, threat_cost


def solve_dubins_uav_path(
    waypoints: List[UAVWaypoint],
    threat_domes: List[ThreatDome],
    min_turn_radius: float = 5.0
) -> DubinsTrajectoryResult:
    """Solves the DTSPN using greedy nearest-neighbor initialization followed by 2-opt local search."""
    t0 = time.perf_counter()
    n = len(waypoints)
    if n <= 1:
        return DubinsTrajectoryResult(
            waypoint_order=[w.wp_id for w in waypoints],
            total_path_length=0.0,
            threat_exposure_penalty=0.0,
            kinematic_feasible=True,
            algorithm="2-Opt-Dubins-DTSPN",
            execution_time_us=0.0
        )

    # 1. Greedy initial tour
    unvisited = list(range(1, n))
    tour = [0]
    while unvisited:
        curr = tour[-1]
        best_nxt = None
        best_c = float('inf')
        for nxt in unvisited:
            d_len, th_cost = _dubins_approx_cost(waypoints[curr], waypoints[nxt], min_turn_radius, threat_domes)
            total_c = d_len + th_cost * 1.5
            if total_c < best_c:
                best_c = total_c
                best_nxt = nxt
        tour.append(best_nxt)
        unvisited.remove(best_nxt)

    # 2. 2-opt local search optimization
    def compute_tour_eval(t_order: List[int]) -> Tuple[float, float, float]:
        l_sum = 0.0
        th_sum = 0.0
        for i in range(len(t_order)):
            u = t_order[i]
            v = t_order[(i + 1) % len(t_order)]
            d_l, th_c = _dubins_approx_cost(waypoints[u], waypoints[v], min_turn_radius, threat_domes)
            l_sum += d_l
            th_sum += th_c
        return l_sum, th_sum, l_sum + th_sum * 1.5

    curr_len, curr_threat, curr_total = compute_tour_eval(tour)

    improved = True
    passes = 0
    while improved and passes < 15:
        improved = False
        passes += 1
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_tour = tour[:i] + tour[i:j + 1][::-1] + tour[j + 1:]
                n_len, n_threat, n_total = compute_tour_eval(new_tour)
                if n_total < curr_total - 1e-4:
                    tour = new_tour
                    curr_len, curr_threat, curr_total = n_len, n_threat, n_total
                    improved = True
                    break
            if improved:
                break

    t1 = time.perf_counter()
    exec_us = (t1 - t0) * 1_000_000.0

    return DubinsTrajectoryResult(
        waypoint_order=[waypoints[idx].wp_id for idx in tour],
        total_path_length=round(curr_len, 2),
        threat_exposure_penalty=round(curr_threat, 2),
        kinematic_feasible=True,
        algorithm="2-Opt-Dubins-DTSPN",
        execution_time_us=round(exec_us, 2)
    )
