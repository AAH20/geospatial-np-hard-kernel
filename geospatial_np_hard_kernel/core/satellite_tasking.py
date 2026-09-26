"""Agile Earth Observation Satellite Scheduling (AEOSSP) Solver.

Solves the NP-hard agile satellite constellation tasking problem with dynamic attitude slewing transitions,
time-window constraints, and high-value priority collection maximization.
Uses time-indexed insertion heuristics with certified angular stabilization clearance.
"""
import math
import time
from typing import List, Tuple
from .models import ObservationTarget, ConstellationScheduleResult


def _compute_slew_time(t1: ObservationTarget, t2: ObservationTarget, slew_rate: float) -> float:
    """Computes necessary slewing and stabilization time between two observation targets."""
    # Approximate angular displacement from ground coordinates (in degrees)
    dist_deg = math.hypot(t2.x - t1.x, t2.y - t1.y) * 0.1  # scaling factor
    stabilization_buffer = 1.0  # seconds
    return (dist_deg / max(0.1, slew_rate)) + stabilization_buffer


def solve_satellite_tasking(
    targets: List[ObservationTarget],
    slew_rate_deg_per_sec: float = 3.0,
    mission_duration: float = 1000.0
) -> ConstellationScheduleResult:
    """Solves AEOSSP maximizing priority collection while guaranteeing slew kinematic feasibility."""
    t0 = time.perf_counter()
    if not targets:
        return ConstellationScheduleResult(
            scheduled_tasks=[],
            total_collected_priority=0.0,
            capacity_utilization_pct=0.0,
            slew_transition_certified=True,
            algorithm="Dynamic-Slew-AEOSSP",
            execution_time_us=0.0
        )

    # Sort targets by value density: priority / duration descending, then earliest window start
    sorted_targets = sorted(
        targets,
        key=lambda t: (t.priority / max(0.1, t.duration), -t.window_start),
        reverse=True
    )

    scheduled: List[Tuple[ObservationTarget, float, float]] = []  # (target, start_t, end_t)
    total_priority = 0.0

    for cand in sorted_targets:
        # Try to insert cand into existing schedule chronologically
        best_insert_pos = None
        best_start_t = None

        # Check all possible insertion positions: 0 .. len(scheduled)
        for pos in range(len(scheduled) + 1):
            prev_task = scheduled[pos - 1] if pos > 0 else None
            next_task = scheduled[pos] if pos < len(scheduled) else None

            # Earliest start time based on window and previous task slew
            if prev_task:
                slew_needed = _compute_slew_time(prev_task[0], cand, slew_rate_deg_per_sec)
                earliest_start = max(cand.window_start, prev_task[2] + slew_needed)
            else:
                earliest_start = cand.window_start

            cand_end = earliest_start + cand.duration

            # Check validity against window end
            if cand_end > cand.window_end:
                continue

            # Check transition to next task
            if next_task:
                slew_to_next = _compute_slew_time(cand, next_task[0], slew_rate_deg_per_sec)
                if cand_end + slew_to_next > next_task[1]:
                    continue

            # Feasible insertion point found
            best_insert_pos = pos
            best_start_t = earliest_start
            break

        if best_insert_pos is not None:
            scheduled.insert(best_insert_pos, (cand, best_start_t, best_start_t + cand.duration))
            total_priority += cand.priority

    # Verify kinematic slew transitions
    certified = True
    for i in range(len(scheduled) - 1):
        t_curr, _, end_curr = scheduled[i]
        t_nxt, start_nxt, _ = scheduled[i + 1]
        needed_slew = _compute_slew_time(t_curr, t_nxt, slew_rate_deg_per_sec)
        if end_curr + needed_slew > start_nxt + 1e-4:
            certified = False
            break

    total_active_time = sum(dur for _, start, end in scheduled for dur in [end - start])
    utilization_pct = min(100.0, (total_active_time / max(1.0, mission_duration)) * 100.0)

    t1 = time.perf_counter()
    exec_us = (t1 - t0) * 1_000_000.0

    return ConstellationScheduleResult(
        scheduled_tasks=[(t.target_id, round(s, 2), round(e, 2)) for t, s, e in scheduled],
        total_collected_priority=round(total_priority, 2),
        capacity_utilization_pct=round(utilization_pct, 2),
        slew_transition_certified=certified,
        algorithm="Dynamic-Slew-AEOSSP",
        execution_time_us=round(exec_us, 2)
    )
