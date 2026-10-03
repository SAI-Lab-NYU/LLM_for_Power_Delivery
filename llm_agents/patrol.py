"""Reveal-efficient patrol planning.

Key insight: Area.knowledge_state depends ONLY on node/line pixels. Patrolling
the geometric centre of a 17x21 unknown rectangle wastes most of the horizon.
We instead do greedy max-coverage over the unknown *network* pixels, read from
the simulator's own element table.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from llm_agents.topology import Topology


def unknown_network_pixels(state: Dict[str, Any], topo: Topology) -> List[Tuple[int, int]]:
    rows = (((state.get("status_tables") or {}).get("elements") or {}).get("rows")) or []
    px: List[Tuple[int, int]] = []
    for row in rows:
        if len(row) < 3:
            continue
        typ, name, knowledge = row[0], row[1], str(row[2]).lower()
        if knowledge == "known":
            continue
        if typ == "Node" and name in topo.node_pos:
            px.append(topo.node_pos[name])
        elif typ == "Line" and name in topo.line_pixels:
            px.extend(topo.line_pixels[name])
    return sorted(set(px))


def unknown_areas(state: Dict[str, Any]) -> List[Dict[str, str]]:
    rows = (((state.get("status_tables") or {}).get("areas") or {}).get("rows")) or []
    out = []
    for row in rows:
        if len(row) < 6:
            continue
        out.append({"area": row[0], "nodes": row[1], "lines": row[2],
                    "boundary_switches": row[3], "knowledge": row[4],
                    "energizable": row[5]})
    return out


def coverage_route(start: Tuple[int, int], targets: List[Tuple[int, int]],
                   radius: int, topo: Topology, max_waypoints: int = 5
                   ) -> Tuple[List[List[int]], int]:
    """Greedy set-cover route. Returns (waypoints, covered_count)."""
    if not targets or radius < 0:
        return [], 0
    remaining = set(targets)
    here = tuple(start)
    route: List[List[int]] = []

    # Candidate stand-points = the target pixels themselves (a superset of good centres).
    while remaining and len(route) < max_waypoints:
        best, best_gain, best_cost = None, 0, None
        for cand in list(remaining):
            gain = sum(1 for p in remaining
                       if max(abs(p[0] - cand[0]), abs(p[1] - cand[1])) <= radius)
            cost = abs(cand[0] - here[0]) + abs(cand[1] - here[1]) + 1
            score = gain / cost
            if best is None or score > best_gain / max(best_cost, 1):
                best, best_gain, best_cost = cand, gain, cost
        if best is None or best_gain == 0:
            break
        route.append(topo.clamp(best))
        remaining = {p for p in remaining
                     if max(abs(p[0] - best[0]), abs(p[1] - best[1])) > radius}
        here = best
    return route, len(targets) - len(remaining)