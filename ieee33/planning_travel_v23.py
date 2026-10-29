"""Read-only exact road travel matrices shared by API and tests."""

from __future__ import annotations

import copy
from typing import Any


def planning_travel_times_for(
    source: Any,
    locations: dict[str, tuple[int, int]],
    unit_names: list[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """Calculate exact road matrices without changing live mobile units."""

    matrices: dict[str, dict[str, dict[str, float]]] = {}
    for unit_name in unit_names:
        if unit_name not in source.mobile_units:
            raise KeyError(f"Unknown mobile unit: {unit_name}")
        unit = copy.deepcopy(source.mobile_units[unit_name])
        matrices[unit_name] = {}
        for origin_name, origin in locations.items():
            unit.position = origin
            unit.planned_path.clear()
            unit.waypoints.clear()
            matrices[unit_name][origin_name] = {}
            for destination_name, destination in locations.items():
                path = list(unit.preview_move(destination))
                route = [origin, *path]
                duration = sum(
                    unit.movement_cost(first, second) * source.minutes_per_step
                    for first, second in zip(route, route[1:])
                )
                matrices[unit_name][origin_name][destination_name] = round(
                    duration, 6
                )
    return matrices
