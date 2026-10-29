"""Build public, finite patrol macro tasks from dynamic Pixel knowledge."""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from integration.exact_open_set_cover_route import OpenSetCoverRouteResult


Position = tuple[int, int]


_EXACT_ROUTE_CACHE: dict[
    tuple[int, tuple[Position, ...], int, float], Any
] = {}


def _position_key(position: Position) -> tuple[int, int]:
    return position[0], position[1]


def _neighbors(
    positions: set[Position],
) -> dict[Position, tuple[Position, ...]]:
    adjacency: dict[Position, tuple[Position, ...]] = {}
    for x, y in positions:
        adjacent = (
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        )
        adjacency[(x, y)] = tuple(sorted(
            (point for point in adjacent if point in positions),
            key=_position_key,
        ))
    return adjacency


def _connected_components(
    adjacency: dict[Position, tuple[Position, ...]],
) -> list[set[Position]]:
    components: list[set[Position]] = []
    unseen = set(adjacency)
    while unseen:
        start = min(unseen, key=_position_key)
        component = {start}
        queue = deque([start])
        unseen.remove(start)
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return components


def _base_edge_cost(source: Any, first: Position, second: Position) -> float:
    first_level = source.graph.get_pixel(first).road_level
    second_level = source.graph.get_pixel(second).road_level
    road_level = min(first_level, second_level)
    multiplier = 1.5 if road_level >= 2 else (1.0 if road_level >= 1 else 0.0)
    return 1.0 / multiplier if multiplier > 0.0 else 1.0


def _shortest_path_tree(
    source: Any,
    adjacency: dict[Position, tuple[Position, ...]],
    component: set[Position],
    targets: set[Position],
) -> dict[Position, set[Position]]:
    """Return a deterministic shortest-path tree connecting all targets."""

    root = min(targets, key=_position_key)
    distances = {root: 0.0}
    previous: dict[Position, Position] = {}
    queue: list[tuple[float, Position]] = [(0.0, root)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances[current] + 1e-12:
            continue
        for neighbor in adjacency[current]:
            if neighbor not in component:
                continue
            candidate = distance + _base_edge_cost(source, current, neighbor)
            incumbent = distances.get(neighbor, float("inf"))
            if candidate + 1e-12 < incumbent:
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))

    tree: dict[Position, set[Position]] = {root: set()}
    for target in sorted(targets, key=_position_key):
        if target not in distances:
            raise RuntimeError(f"Patrol target {target} is disconnected from {root}.")
        current = target
        tree.setdefault(current, set())
        while current != root:
            parent = previous[current]
            tree.setdefault(parent, set()).add(current)
            tree.setdefault(current, set()).add(parent)
            current = parent
    return tree


def _tree_distances(
    source: Any,
    tree: dict[Position, set[Position]],
    start: Position,
) -> tuple[dict[Position, float], dict[Position, Position]]:
    distances = {start: 0.0}
    parent: dict[Position, Position] = {}
    stack = [start]
    while stack:
        current = stack.pop()
        for neighbor in sorted(tree[current], key=_position_key, reverse=True):
            if neighbor == parent.get(current):
                continue
            parent[neighbor] = current
            distances[neighbor] = (
                distances[current] + _base_edge_cost(source, current, neighbor)
            )
            stack.append(neighbor)
    return distances, parent


def _farthest(
    distances: dict[Position, float],
) -> Position:
    return min(
        distances,
        key=lambda point: (-distances[point], _position_key(point)),
    )


def _tree_diameter_path(
    source: Any,
    tree: dict[Position, set[Position]],
) -> list[Position]:
    start = min(tree, key=_position_key)
    first_distances, _ = _tree_distances(source, tree, start)
    first = _farthest(first_distances)
    second_distances, parent = _tree_distances(source, tree, first)
    second = _farthest(second_distances)
    reverse_path = [second]
    current = second
    while current != first:
        current = parent[current]
        reverse_path.append(current)
    reverse_path.reverse()
    return reverse_path


def _tree_path(
    source: Any,
    tree: dict[Position, set[Position]],
    start: Position,
    end: Position,
) -> list[Position]:
    _, parent = _tree_distances(source, tree, start)
    reverse_path = [end]
    current = end
    while current != start:
        current = parent[current]
        reverse_path.append(current)
    reverse_path.reverse()
    return reverse_path


def _coverage_walk(
    tree: dict[Position, set[Position]],
    main_path: list[Position],
) -> list[Position]:
    """Traverse side branches twice and the diameter path once."""

    walk = [main_path[0]]

    def visit_branch(parent: Position, current: Position) -> None:
        walk.append(current)
        for neighbor in sorted(tree[current], key=_position_key):
            if neighbor != parent:
                visit_branch(current, neighbor)
        walk.append(parent)

    for index, current in enumerate(main_path):
        previous = main_path[index - 1] if index > 0 else None
        following = main_path[index + 1] if index + 1 < len(main_path) else None
        for neighbor in sorted(tree[current], key=_position_key):
            if neighbor not in {previous, following}:
                visit_branch(current, neighbor)
        if following is not None:
            walk.append(following)
    return walk


def _route_cost(source: Any, route: Iterable[Position]) -> float:
    positions = list(route)
    return sum(
        _base_edge_cost(source, first, second)
        for first, second in zip(positions, positions[1:])
    )


def _unit_route_time_min(
    source: Any,
    unit: Any,
    route: list[Position],
    continuous_time: bool = False,
) -> float:
    raw_time = _route_cost(source, route) / float(unit.move_speed) \
        * float(source.minutes_per_step)
    if continuous_time:
        return raw_time
    frame_min = source.minutes_per_step / source.physics_frames_per_step
    return max(frame_min, math.ceil((raw_time - 1e-12) / frame_min) * frame_min)


def _travel_to_entry_min(source: Any, unit: Any, entry: Position) -> float:
    path = list(unit.preview_move(entry))
    route = [tuple(unit.position), *path]
    return _route_cost(source, route) / float(unit.move_speed) \
        * float(source.minutes_per_step)


def _expected_detection_time_min(
    source: Any,
    unit: Any,
    route: list[Position],
    electrical_positions: set[Position],
) -> float:
    """Expected first-observation time for a uniform fault Pixel on a line."""

    if not electrical_positions:
        return 0.0
    elapsed = [0.0]
    for first, second in zip(route, route[1:]):
        elapsed.append(elapsed[-1] + (
            _base_edge_cost(source, first, second)
            / float(unit.move_speed)
            * float(source.minutes_per_step)
        ))
    detection_times = []
    inspection_range = int(unit.inspection_range)
    for electrical in electrical_positions:
        ex, ey = electrical
        first_seen = next((
            elapsed[index]
            for index, (x, y) in enumerate(route)
            if max(abs(x - ex), abs(y - ey)) <= inspection_range
        ), elapsed[-1])
        detection_times.append(first_seen)
    return sum(detection_times) / len(detection_times)


def _unit_profiles(
    source: Any,
    route: list[Position],
    continuous_time: bool = False,
) -> dict[str, dict[str, Any]]:
    entry = route[0]
    profiles: dict[str, dict[str, Any]] = {}
    for unit in source.mobile_units.values():
        if unit.unit_type not in {"RC", "Scott"}:
            continue
        travel_min = _travel_to_entry_min(source, unit, entry)
        patrol_min = _unit_route_time_min(
            source, unit, route, continuous_time=continuous_time
        )
        profiles[unit.name] = {
            "available_now": unit.state not in {
                "Moving", "Analyzing", "Repairing", "OperatingSwitch",
            },
            "current_position": list(unit.position),
            "inspection_range_pixels": int(unit.inspection_range),
            "travel_to_entry_min": round(travel_min, 6),
            "patrol_duration_min": round(patrol_min, 6),
            "total_from_current_min": round(travel_min + patrol_min, 6),
            "patrol_command_steps": max(
                1, math.ceil((patrol_min - 1e-12) / source.minutes_per_step)
            ),
        }
    return profiles


def _observation_position(
    source: Any,
    electrical_position: Position,
    inspection_range: int,
) -> tuple[Position, str]:
    """Map one electrical Pixel to a nearby road/service observation Pixel."""

    candidates: list[tuple[int, int, int, Position, str]] = []
    ex, ey = electrical_position
    for x in range(
        max(source.graph.x_min, ex - inspection_range),
        min(source.graph.x_max, ex + inspection_range) + 1,
    ):
        for y in range(
            max(source.graph.y_min, ey - inspection_range),
            min(source.graph.y_max, ey + inspection_range) + 1,
        ):
            distance = max(abs(x - ex), abs(y - ey))
            if distance > inspection_range:
                continue
            pixel = source.graph.get_pixel((x, y))
            explicit_roads = {
                name for name in pixel.road_names
                if name != "__electrical_network__"
            }
            if explicit_roads:
                category = 0
                kind = "explicit_road"
            elif pixel.road_level >= 1:
                category = 1
                kind = "electrical_service_road"
            else:
                continue
            candidates.append((
                category,
                distance,
                -int(pixel.road_level),
                (x, y),
                kind,
            ))
    if not candidates:
        raise RuntimeError(
            f"No observable road/service Pixel for {electrical_position}."
        )
    _, _, _, position, kind = min(candidates)
    return position, kind


def _observation_candidates(
    source: Any,
    electrical_position: Position,
    inspection_range: int,
) -> dict[Position, str]:
    """Return every mobility Pixel that can observe one electrical Pixel."""

    candidates: dict[Position, str] = {}
    ex, ey = electrical_position
    for x in range(
        max(source.graph.x_min, ex - inspection_range),
        min(source.graph.x_max, ex + inspection_range) + 1,
    ):
        for y in range(
            max(source.graph.y_min, ey - inspection_range),
            min(source.graph.y_max, ey + inspection_range) + 1,
        ):
            if max(abs(x - ex), abs(y - ey)) > inspection_range:
                continue
            pixel = source.graph.get_pixel((x, y))
            explicit_roads = {
                name for name in pixel.road_names
                if name != "__electrical_network__"
            }
            if explicit_roads:
                candidates[(x, y)] = "explicit_road"
            elif pixel.road_level >= 1:
                candidates[(x, y)] = "electrical_service_road"
    if not candidates:
        raise RuntimeError(
            f"No observable road/service Pixel for {electrical_position}."
        )
    return candidates


def _coverage_index(
    source: Any,
    required_positions: set[Position],
    inspection_range: int,
) -> tuple[
    dict[Position, set[Position]],
    dict[Position, set[Position]],
    dict[Position, str],
]:
    """Build the many-to-many electrical-Pixel/road-Pixel coverage matrix."""

    covered_by_position: dict[Position, set[Position]] = {}
    candidates_by_target: dict[Position, set[Position]] = {}
    kinds: dict[Position, str] = {}
    for electrical_position in sorted(required_positions, key=_position_key):
        candidates = _observation_candidates(
            source, electrical_position, inspection_range
        )
        candidates_by_target[electrical_position] = set(candidates)
        for position, kind in candidates.items():
            covered_by_position.setdefault(position, set()).add(
                electrical_position
            )
            if kind == "explicit_road":
                kinds[position] = kind
            else:
                kinds.setdefault(position, kind)
    return covered_by_position, candidates_by_target, kinds


def _multi_source_shortest_paths(
    source: Any,
    adjacency: dict[Position, tuple[Position, ...]],
    roots: set[Position],
) -> tuple[dict[Position, float], dict[Position, Position]]:
    """Return shortest mobility paths from the current connected route tree."""

    distances = {root: 0.0 for root in roots}
    previous: dict[Position, Position] = {}
    queue = [(0.0, root) for root in sorted(roots, key=_position_key)]
    heapq.heapify(queue)
    while queue:
        distance, current = heapq.heappop(queue)
        if distance > distances[current] + 1e-12:
            continue
        for neighbor in adjacency[current]:
            candidate = distance + _base_edge_cost(source, current, neighbor)
            incumbent = distances.get(neighbor, float("inf"))
            if candidate + 1e-12 < incumbent:
                distances[neighbor] = candidate
                previous[neighbor] = current
                heapq.heappush(queue, (candidate, neighbor))
    return distances, previous


def _path_to_tree(
    position: Position,
    tree_positions: set[Position],
    previous: dict[Position, Position],
) -> list[Position]:
    """Reconstruct a shortest path from the existing tree to a position."""

    reverse_path = [position]
    current = position
    while current not in tree_positions:
        current = previous[current]
        reverse_path.append(current)
    reverse_path.reverse()
    return reverse_path


def _tree_walk_cost(
    source: Any,
    tree: dict[Position, set[Position]],
) -> float:
    endpoint_pairs = _candidate_endpoint_pairs(source, tree)
    return min(
        _route_cost(
            source,
            _coverage_walk(tree, _tree_path(source, tree, first, second)),
        )
        for first, second in endpoint_pairs
    )


def _path_primitive_cover_tree(
    source: Any,
    adjacency: dict[Position, tuple[Position, ...]],
    unit: Any,
    required_positions: set[Position],
    max_seed_count: int = 8,
) -> tuple[
    dict[Position, set[Position]],
    dict[Position, set[Position]],
    dict[Position, set[Position]],
    dict[Position, str],
]:
    """Build a connected route tree with a route-aware set-cover heuristic.

    Every road Pixel is a candidate observation point and every traversed road
    edge is a path primitive. A shortest path is added according to incremental
    travel cost per newly covered electrical Pixel, so coverage gained while
    moving is credited directly.
    """

    coverage, candidates_by_target, kinds = _coverage_index(
        source, required_positions, int(unit.inspection_range)
    )
    ranked_seeds = sorted(
        coverage,
        key=lambda position: (
            -len(coverage[position]),
            0 if kinds[position] == "explicit_road" else 1,
            -int(source.graph.get_pixel(position).road_level),
            _position_key(position),
        ),
    )[:max_seed_count]
    best: tuple[float, int, tuple[int, int], dict[Position, set[Position]]] | None = None

    for seed in ranked_seeds:
        tree: dict[Position, set[Position]] = {seed: set()}
        tree_positions = {seed}
        covered = set(coverage[seed])
        while not required_positions.issubset(covered):
            distances, previous = _multi_source_shortest_paths(
                source, adjacency, tree_positions
            )
            remaining = required_positions - covered
            choices = []
            for position in coverage:
                if position in tree_positions or position not in distances:
                    continue
                path = _path_to_tree(position, tree_positions, previous)
                path_coverage = set().union(*(
                    coverage.get(point, set()) for point in path
                ))
                gain = len(path_coverage.intersection(remaining))
                if gain == 0:
                    continue
                choices.append((
                    distances[position] / gain,
                    distances[position],
                    0 if kinds[position] == "explicit_road" else 1,
                    -gain,
                    _position_key(position),
                    path,
                ))
            if not choices:
                missing = min(remaining, key=_position_key)
                raise RuntimeError(
                    f"No connected observation route can cover {missing}."
                )
            *_, path = min(choices)
            for first, second in zip(path, path[1:]):
                tree.setdefault(first, set()).add(second)
                tree.setdefault(second, set()).add(first)
            tree_positions.update(path)
            for point in path:
                covered.update(coverage.get(point, set()))

        score = (
            _tree_walk_cost(source, tree),
            len(tree),
            _position_key(seed),
            tree,
        )
        if best is None or score[:3] < best[:3]:
            best = score

    if best is None:
        raise RuntimeError("Cannot build a path-primitive coverage tree.")
    return best[3], coverage, candidates_by_target, kinds


def _exact_open_set_cover_walk(
    source: Any,
    adjacency: dict[Position, tuple[Position, ...]],
    unit: Any,
    required_positions: set[Position],
    time_limit_sec: float = 10.0,
) -> tuple[
    list[Position],
    dict[Position, set[Position]],
    dict[Position, set[Position]],
    dict[Position, str],
    "OpenSetCoverRouteResult",
]:
    """Solve the lower-level open set-covering route MILP exactly."""

    coverage, candidates_by_target, kinds = _coverage_index(
        source, required_positions, int(unit.inspection_range)
    )
    key = (
        id(source.graph),
        tuple(sorted(required_positions, key=_position_key)),
        int(unit.inspection_range),
        float(time_limit_sec),
    )
    result = _EXACT_ROUTE_CACHE.get(key)
    if result is None:
        from integration.exact_open_set_cover_route import (
            OpenSetCoverRouteResult,
            solve_open_set_cover_route,
        )

        warm_tree, _, _, _ = _path_primitive_cover_tree(
            source, adjacency, unit, required_positions
        )
        warm_routes = [
            _coverage_walk(
                warm_tree,
                _tree_path(source, warm_tree, first, second),
            )
            for first, second in _candidate_endpoint_pairs(source, warm_tree)
        ]
        initial_route = min(
            warm_routes,
            key=lambda route: (_route_cost(source, route), tuple(route)),
        )
        try:
            result = solve_open_set_cover_route(
                adjacency,
                lambda first, second: _base_edge_cost(source, first, second),
                candidates_by_target,
                initial_route=initial_route,
                time_limit_sec=time_limit_sec,
                output_flag=False,
            )
        except RuntimeError as error:
            if "has no incumbent" not in str(error):
                raise
            route = tuple(initial_route)
            result = OpenSetCoverRouteResult(
                route=route,
                objective=_route_cost(source, list(route)),
                selected_nodes=tuple(dict.fromkeys(route)),
                traversed_arc_count=max(0, len(route) - 1),
                solver_status=9,
                solver_runtime_sec=float(time_limit_sec),
                solver_mip_gap=1.0,
                solver_objective_bound=0.0,
                used_feasible_fallback=True,
                fallback_reason=str(error),
            )
        if len(_EXACT_ROUTE_CACHE) >= 256:
            _EXACT_ROUTE_CACHE.pop(next(iter(_EXACT_ROUTE_CACHE)))
        _EXACT_ROUTE_CACHE[key] = result
    return (
        [tuple(point) for point in result.route],
        coverage,
        candidates_by_target,
        kinds,
        result,
    )


def _route_observation_assignment(
    source: Any,
    unit: Any,
    route: list[Position],
    required_positions: set[Position],
) -> tuple[dict[Position, Position], dict[Position, str]]:
    """Assign each electrical Pixel to its first actual route observation."""

    mapping: dict[Position, Position] = {}
    kinds: dict[Position, str] = {}
    for electrical_position in sorted(required_positions, key=_position_key):
        candidates = _observation_candidates(
            source, electrical_position, int(unit.inspection_range)
        )
        observed_from = next(
            (position for position in route if position in candidates), None
        )
        if observed_from is None:
            raise RuntimeError(
                f"Generated patrol route does not cover {electrical_position}."
            )
        mapping[electrical_position] = observed_from
        kinds[electrical_position] = candidates[observed_from]
    return mapping, kinds


def _unit_observation_targets(
    source: Any,
    unit: Any,
    required_positions: set[Position],
) -> tuple[set[Position], dict[Position, Position], dict[Position, str]]:
    mapping: dict[Position, Position] = {}
    kinds: dict[Position, str] = {}
    for electrical_position in sorted(required_positions, key=_position_key):
        observation_position, kind = _observation_position(
            source,
            electrical_position,
            int(unit.inspection_range),
        )
        mapping[electrical_position] = observation_position
        kinds[electrical_position] = kind
    return set(mapping.values()), mapping, kinds


def _candidate_endpoint_pairs(
    source: Any,
    tree: dict[Position, set[Position]],
    max_unordered_pairs: int = 2,
) -> list[tuple[Position, Position]]:
    leaves = sorted(
        (point for point, neighbors in tree.items() if len(neighbors) <= 1),
        key=_position_key,
    )
    if len(tree) == 1:
        only = next(iter(tree))
        return [(only, only)]
    if len(leaves) < 2:
        diameter = _tree_diameter_path(source, tree)
        return [(diameter[0], diameter[-1])]

    ranked = []
    for index, first in enumerate(leaves):
        distances, _ = _tree_distances(source, tree, first)
        for second in leaves[index + 1:]:
            ranked.append((
                -float(distances[second]),
                _position_key(first),
                _position_key(second),
                first,
                second,
            ))
    ranked.sort()
    return [
        (first, second)
        for _, _, _, first, second in ranked[:max_unordered_pairs]
    ]


def _pareto_endpoint_nodes(
    source: Any,
    unit: Any,
    tree: dict[Position, set[Position]],
    max_nodes: int = 16,
) -> list[Position]:
    """Select diverse leaf and internal endpoints for Pareto route generation."""

    leaves = sorted(
        (point for point, values in tree.items() if len(values) <= 1),
        key=_position_key,
    )
    nearest = sorted(
        tree,
        key=lambda point: (
            _travel_to_entry_min(source, unit, point),
            _position_key(point),
        ),
    )[:5]
    extremes = []
    for key in (
        lambda point: (point[0], point[1]),
        lambda point: (-point[0], point[1]),
        lambda point: (point[1], point[0]),
        lambda point: (-point[1], point[0]),
    ):
        extremes.append(min(tree, key=key))
    diameter = _tree_diameter_path(source, tree)
    ordered = [
        *leaves,
        diameter[0],
        diameter[-1],
        *nearest,
        *extremes,
    ]
    unique = list(dict.fromkeys(ordered))
    return unique[:max_nodes]


def _route_pareto_metrics(
    source: Any,
    unit: Any,
    route: list[Position],
    line_targets: dict[str, set[Position]],
    continuous_time: bool,
) -> tuple[float, float, float]:
    patrol_min = _unit_route_time_min(
        source, unit, route, continuous_time=continuous_time
    )
    approach_min = _travel_to_entry_min(source, unit, route[0])
    detection_values = [
        _expected_detection_time_min(source, unit, route, targets)
        for targets in line_targets.values() if targets
    ]
    mean_detection_min = (
        sum(detection_values) / len(detection_values)
        if detection_values else 0.0
    )
    return approach_min + patrol_min, patrol_min, mean_detection_min


def _dominates(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
) -> bool:
    return (
        all(a <= b + 1e-9 for a, b in zip(first, second))
        and any(a < b - 1e-9 for a, b in zip(first, second))
    )


def _pareto_route_candidates(
    source: Any,
    unit: Any,
    tree: dict[Position, set[Position]],
    exact_route: list[Position],
    line_targets: dict[str, set[Position]],
    continuous_time: bool,
    oscrp_origin: str = "oscrp_milp_incumbent",
    min_routes: int = 3,
    max_routes: int = 5,
) -> list[dict[str, Any]]:
    """Return first-front routes plus diverse routes from later Pareto fronts."""

    raw: dict[tuple[Position, ...], str] = {
        tuple(exact_route): oscrp_origin,
    }
    if len(exact_route) > 1:
        raw[tuple(reversed(exact_route))] = (
            "path_primitive_fallback_reverse"
            if oscrp_origin == "path_primitive_feasible_fallback"
            else "oscrp_milp_reverse"
        )
    endpoints = _pareto_endpoint_nodes(source, unit, tree)
    legacy_alternatives = []
    for pair_index, (first, second) in enumerate(
        _candidate_endpoint_pairs(source, tree), start=1
    ):
        forward = _coverage_walk(
            tree, _tree_path(source, tree, first, second)
        )
        orientations = [("forward", forward)]
        if len(forward) > 1:
            orientations.append(("reverse", list(reversed(forward))))
        for direction, route in orientations:
            total = _travel_to_entry_min(source, unit, route[0]) + (
                _unit_route_time_min(
                    source,
                    unit,
                    route,
                    continuous_time=continuous_time,
                )
            )
            legacy_alternatives.append((
                total,
                len(route),
                f"{unit.name}:pair{pair_index}:{direction}",
                tuple(route),
            ))
    reference_route = min(legacy_alternatives)[-1]
    raw.setdefault(reference_route, "path_primitive_reference")
    for index, first in enumerate(endpoints):
        for second in endpoints[index + 1:]:
            route = _coverage_walk(
                tree, _tree_path(source, tree, first, second)
            )
            raw.setdefault(tuple(route), "endpoint_variant")
            raw.setdefault(tuple(reversed(route)), "endpoint_variant")

    candidates = []
    for route_tuple, origin in raw.items():
        route = list(route_tuple)
        metrics = _route_pareto_metrics(
            source,
            unit,
            route,
            line_targets,
            continuous_time,
        )
        candidates.append({
            "route": route,
            "origin": origin,
            "metrics": metrics,
        })

    remaining = list(candidates)
    ranked: list[dict[str, Any]] = []
    rank = 1
    while remaining:
        front = [
            row for row in remaining
            if not any(
                _dominates(other["metrics"], row["metrics"])
                for other in remaining if other is not row
            )
        ]
        for row in front:
            row["pareto_rank"] = rank
            row["pareto_nondominated"] = rank == 1
        ranked.extend(sorted(
            front,
            key=lambda row: (
                row["metrics"],
                _position_key(row["route"][0]),
                _position_key(row["route"][-1]),
            ),
        ))
        front_ids = {id(row) for row in front}
        remaining = [row for row in remaining if id(row) not in front_ids]
        rank += 1

    selected: list[dict[str, Any]] = []
    first_front = [row for row in ranked if row["pareto_rank"] == 1]
    for metric_index in range(3):
        if first_front:
            row = min(
                first_front,
                key=lambda value: (
                    value["metrics"][metric_index],
                    value["metrics"],
                ),
            )
            if row not in selected:
                selected.append(row)

    def endpoint_distance(row: dict[str, Any]) -> float:
        if not selected:
            return float("inf")
        entry, exit_ = row["route"][0], row["route"][-1]
        return min(
            abs(entry[0] - other["route"][0][0])
            + abs(entry[1] - other["route"][0][1])
            + abs(exit_[0] - other["route"][-1][0])
            + abs(exit_[1] - other["route"][-1][1])
            for other in selected
        )

    target_count = min(
        len(ranked),
        max(min_routes, min(max_routes, len(first_front))),
    )
    while len(selected) < target_count:
        eligible = [row for row in ranked if row not in selected]
        if not eligible:
            break
        if len(selected) < min_routes:
            best_rank = min(row["pareto_rank"] for row in eligible)
            eligible = [
                row for row in eligible if row["pareto_rank"] == best_rank
            ]
        row = min(
            eligible,
            key=lambda value: (
                value["pareto_rank"],
                -endpoint_distance(value),
                value["metrics"],
            ),
        )
        selected.append(row)

    reference_candidate = next(
        row for row in ranked if tuple(row["route"]) == reference_route
    )
    if reference_candidate not in selected:
        if len(selected) < max_routes:
            selected.append(reference_candidate)
        else:
            selected[-1] = reference_candidate

    return selected[:max_routes]


def _line_coverage_rows(
    source: Any,
    area: Any,
    unit_names: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for line_name in area.line_names:
        line = source.electrical_lines[line_name]
        remaining = [
            pixel.position for pixel in line.effective_pixels
            if not pixel.is_inspected(source.event_id)
        ]
        observations: dict[str, list[list[int]]] = {}
        candidate_observations: dict[str, list[list[int]]] = {}
        fallback_by_unit: dict[str, bool] = {}
        for unit_name in unit_names:
            unit = source.mobile_units[unit_name]
            positions = set()
            all_candidates = set()
            fallback = False
            for position in remaining:
                observed_from, kind = _observation_position(
                    source, position, int(unit.inspection_range)
                )
                positions.add(observed_from)
                fallback = fallback or kind != "explicit_road"
                all_candidates.update(_observation_candidates(
                    source, position, int(unit.inspection_range)
                ))
            observations[unit_name] = [
                list(point) for point in sorted(positions, key=_position_key)
            ]
            candidate_observations[unit_name] = [
                list(point)
                for point in sorted(all_candidates, key=_position_key)
            ]
            fallback_by_unit[unit_name] = fallback
        rows.append({
            "line": line_name,
            "knowledge": line.knowledge_state(source.event_id),
            "covered": not remaining,
            "remaining_electrical_pixels": [
                list(point) for point in sorted(remaining, key=_position_key)
            ],
            "observation_pixels_by_unit": observations,
            "candidate_observation_pixels_by_unit": candidate_observations,
            "uses_service_road_fallback_by_unit": fallback_by_unit,
        })
    return rows


def build_patrol_macro_payload(
    source: Any,
    continuous_time: bool = False,
    strategy: str = "path_primitive_cover",
    lower_level_time_limit_sec: float = 10.0,
) -> dict[str, Any]:
    """Convert current public Area/Pixel knowledge into finite patrol tasks.

    A task covers one connected remaining component of an Area's electrical
    skeleton. The default strategy selects a connected road path whose moving
    sensor footprint covers every required Pixel. The legacy one-point mapping
    remains available as ``representative_points`` for controlled comparison.
    Hidden fault state is never read or returned.
    """

    if hasattr(source, "segment_patrol_payload"):
        return source.segment_patrol_payload(continuous_time, strategy, lower_level_time_limit_sec)
    if source.graph is None:
        return {"areas": [], "tasks": []}
    if strategy not in {
        "exact_open_set_cover_route",
        "path_primitive_cover",
        "representative_points",
    }:
        raise ValueError(f"Unsupported patrol planning strategy: {strategy}")

    area_rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    patrol_units = [
        unit for unit in source.mobile_units.values()
        if unit.unit_type in {"RC", "Scott"}
    ]
    patrol_unit_names = sorted(unit.name for unit in patrol_units)
    mobility_positions = {
        position for position, pixel in source.graph.pixels.items()
        if pixel.road_level >= 1
    }
    mobility_adjacency = _neighbors(mobility_positions)
    for area_name, area in sorted(source.areas.items()):
        area_positions = {pixel.position for pixel in area.pixels}
        area_load_nodes = [
            node
            for node in source.nodes.values()
            if node.area_name == area_name and node.has_load
        ]
        unknown_positions = {
            pixel.position
            for pixel in area.pixels
            if not pixel.is_inspected(source.event_id)
        }
        adjacency = _neighbors(area_positions)
        components = [
            component
            for component in _connected_components(adjacency)
            if component.intersection(unknown_positions)
        ]
        group_ids: list[str] = []
        for component_index, component in enumerate(components, start=1):
            targets = component.intersection(unknown_positions)
            group_id = f"patrol:{area_name}:component:{component_index}"
            group_ids.append(group_id)
            component_line_names = sorted({
                line_name
                for position in targets
                for line_name in source.graph.get_pixel(position).line_names
                if source.electrical_lines[line_name].area_name == area_name
            })
            line_targets_by_name = {
                line_name: {
                    pixel.position
                    for pixel in source.electrical_lines[line_name].effective_pixels
                    if pixel.position in targets
                }
                for line_name in component_line_names
            }
            for unit in patrol_units:
                exact_result: Any | None = None
                route_metadata: dict[tuple[Position, ...], dict[str, Any]] = {}
                if strategy == "representative_points":
                    observation_targets, _, _ = _unit_observation_targets(
                        source, unit, targets
                    )
                    tree = _shortest_path_tree(
                        source,
                        mobility_adjacency,
                        mobility_positions,
                        observation_targets,
                    )
                    coverage, candidates_by_target, _ = _coverage_index(
                        source, targets, int(unit.inspection_range)
                    )
                    endpoint_routes = []
                elif strategy == "path_primitive_cover":
                    tree, coverage, candidates_by_target, _ = (
                        _path_primitive_cover_tree(
                            source,
                            mobility_adjacency,
                            unit,
                            targets,
                        )
                    )
                    endpoint_routes = []
                else:
                    (
                        exact_route,
                        coverage,
                        candidates_by_target,
                        _,
                        exact_result,
                    ) = _exact_open_set_cover_walk(
                        source,
                        mobility_adjacency,
                        unit,
                        targets,
                        time_limit_sec=lower_level_time_limit_sec,
                    )
                    pareto_tree, _, _, _ = _path_primitive_cover_tree(
                        source,
                        mobility_adjacency,
                        unit,
                        targets,
                    )
                    pareto_routes = _pareto_route_candidates(
                        source,
                        unit,
                        pareto_tree,
                        exact_route,
                        line_targets_by_name,
                        continuous_time,
                        oscrp_origin=(
                            "path_primitive_feasible_fallback"
                            if exact_result.used_feasible_fallback
                            else "oscrp_milp_incumbent"
                        ),
                    )
                    endpoint_routes = [
                        (
                            row["route"][0],
                            row["route"][-1],
                            row["route"],
                        )
                        for row in pareto_routes
                    ]
                    route_metadata = {
                        tuple(row["route"]): row for row in pareto_routes
                    }

                if strategy != "exact_open_set_cover_route":
                    endpoint_pairs = _candidate_endpoint_pairs(source, tree)
                    endpoint_routes = [
                        (
                            first,
                            second,
                            _coverage_walk(
                                tree,
                                _tree_path(source, tree, first, second),
                            ),
                        )
                        for first, second in endpoint_pairs
                    ]

                for pair_index, (first, second, forward_route) in enumerate(
                    endpoint_routes, start=1
                ):
                    orientations = [
                        (
                            f"pareto{pair_index}",
                            forward_route,
                        )
                    ] if strategy == "exact_open_set_cover_route" else [
                        ("forward", forward_route)
                    ]
                    if (
                        strategy != "exact_open_set_cover_route"
                        and len(forward_route) > 1
                    ):
                        orientations.append((
                            "reverse", list(reversed(forward_route))
                        ))
                    for direction, route in orientations:
                        candidate_metadata = route_metadata.get(
                            tuple(route), {}
                        )
                        is_milp_incumbent = candidate_metadata.get("origin") == (
                            "oscrp_milp_incumbent"
                        )
                        observation_map, observation_kinds = (
                            _route_observation_assignment(
                                source, unit, route, targets
                            )
                        )
                        line_observations: dict[str, list[list[int]]] = {}
                        for line_name in component_line_names:
                            line = source.electrical_lines[line_name]
                            positions = {
                                observation_map[pixel.position]
                                for pixel in line.effective_pixels
                                if pixel.position in targets
                            }
                            line_observations[line_name] = [
                                list(point)
                                for point in sorted(positions, key=_position_key)
                            ]
                        orientation = (
                            f"{unit.name}:pair{pair_index}:{direction}"
                        )
                        profiles = _unit_profiles(
                            source, route, continuous_time=continuous_time
                        )
                        profile = {unit.name: profiles[unit.name]}
                        approach = [
                            tuple(unit.position),
                            *[tuple(point) for point in unit.preview_move(route[0])],
                        ]
                        combined_route = [
                            *approach,
                            *(
                                route[1:]
                                if approach and approach[-1] == route[0]
                                else route
                            ),
                        ]
                        patrol_detection_by_line: dict[str, float] = {}
                        start_detection_by_line: dict[str, float] = {}
                        for line_name in component_line_names:
                            line = source.electrical_lines[line_name]
                            line_targets = {
                                pixel.position
                                for pixel in line.effective_pixels
                                if pixel.position in targets
                            }
                            patrol_detection_by_line[line_name] = round(
                                _expected_detection_time_min(
                                    source, unit, route, line_targets
                                ),
                                6,
                            )
                            start_detection_by_line[line_name] = round(
                                _expected_detection_time_min(
                                    source, unit, combined_route, line_targets
                                ),
                                6,
                            )
                        tasks.append({
                            "task_id": f"{group_id}:{orientation}",
                            "alternative_group_id": group_id,
                            "area": area_name,
                            "component_index": component_index,
                            "orientation": orientation,
                            "route_unit": unit.name,
                            "endpoint_pair_index": pair_index,
                            "entry_pixel": list(route[0]),
                            "exit_pixel": list(route[-1]),
                            "remaining_route_pixels": [
                                list(point) for point in route
                            ],
                            "route_edge_count": max(0, len(route) - 1),
                            "coverage_strategy": strategy,
                            "route_candidate_origin": candidate_metadata.get(
                                "origin"
                            ),
                            "pareto_rank": candidate_metadata.get(
                                "pareto_rank"
                            ),
                            "pareto_nondominated": candidate_metadata.get(
                                "pareto_nondominated"
                            ),
                            "pareto_metrics": (
                                {
                                    "total_from_current_min": round(
                                        candidate_metadata["metrics"][0], 6
                                    ),
                                    "patrol_duration_min": round(
                                        candidate_metadata["metrics"][1], 6
                                    ),
                                    "mean_expected_detection_min": round(
                                        candidate_metadata["metrics"][2], 6
                                    ),
                                }
                                if candidate_metadata else None
                            ),
                            "lower_level_route_optimal": (
                                bool(exact_result is not None)
                                and is_milp_incumbent
                                and exact_result.solver_status == 2
                                and exact_result.solver_mip_gap <= 1e-9
                            ),
                            "lower_level_objective": (
                                round(exact_result.objective, 6)
                                if exact_result is not None
                                and is_milp_incumbent else None
                            ),
                            "candidate_route_cost": round(
                                _route_cost(source, route), 6
                            ),
                            "lower_level_solver_runtime_sec": (
                                round(exact_result.solver_runtime_sec, 6)
                                if exact_result is not None else None
                            ),
                            "lower_level_solver_mip_gap": (
                                round(exact_result.solver_mip_gap, 9)
                                if exact_result is not None else None
                            ),
                            "lower_level_solver_objective_bound": (
                                round(exact_result.solver_objective_bound, 6)
                                if exact_result is not None else None
                            ),
                            "lower_level_used_feasible_fallback": (
                                bool(exact_result.used_feasible_fallback)
                                if exact_result is not None else False
                            ),
                            "lower_level_fallback_reason": (
                                exact_result.fallback_reason
                                if exact_result is not None else None
                            ),
                            "candidate_observation_pixel_count": len(coverage),
                            "selected_observation_pixel_count": len(
                                set(observation_map.values())
                            ),
                            "mean_candidate_count_per_target": round(
                                sum(map(len, candidates_by_target.values()))
                                / max(1, len(candidates_by_target)),
                                6,
                            ),
                            "covered_electrical_pixel_count": len(
                                observation_map
                            ),
                            "remaining_unknown_area_pixels": [
                                list(point)
                                for point in sorted(targets, key=_position_key)
                            ],
                            "remaining_unknown_area_pixel_count": len(targets),
                            "required_line_names": component_line_names,
                            "observation_pixels_by_line": line_observations,
                            "expected_patrol_detection_min_by_line": (
                                patrol_detection_by_line
                            ),
                            "expected_start_detection_min_by_line": (
                                start_detection_by_line
                            ),
                            "uses_service_road_fallback": any(
                                kind != "explicit_road"
                                for kind in observation_kinds.values()
                            ),
                            "route_network": (
                                "mobility_grid_with_explicit_and_service_roads"
                            ),
                            "eligible_units": [unit.name],
                            "unit_profiles": profile,
                            "nominal_no_fault_interruption": True,
                        })
        line_coverage = _line_coverage_rows(
            source, area, patrol_unit_names
        )
        area_rows.append({
            "area": area_name,
            "knowledge": area.knowledge_state(source.event_id),
            "electrical_pixel_count": len(area_positions),
            "remaining_unknown_area_pixel_count": len(unknown_positions),
            "required_patrol_groups": group_ids,
            "covered_line_names": [
                row["line"] for row in line_coverage if row["covered"]
            ],
            "remaining_line_names": [
                row["line"] for row in line_coverage if not row["covered"]
            ],
            "line_coverage": line_coverage,
            "boundary_switches": list(area.boundary_switch_names),
            "load_nodes": [node.name for node in area_load_nodes],
            "total_load_kw": round(sum(
                float(node.active_power) for node in area_load_nodes
            ), 6),
            "total_weighted_load_kw": round(sum(
                float(node.active_power) * float(node.load_weight)
                for node in area_load_nodes
            ), 6),
            "energized_weighted_load_kw": round(sum(
                float(node.active_power) * float(node.load_weight)
                for node in area_load_nodes if node.energized
            ), 6),
        })

    return {
        "clock": source.clock_payload(),
        "event_id": int(source.event_id),
        "semantics": {
            "task_granularity": (
                "remaining electrical Pixels covered by mobility path primitives"
            ),
            "route_network": (
                "explicit roads preferred; electrical service roads are fallback"
            ),
            "alternatives": (
                "choose exactly one unit/entry/exit alternative per patrol group"
            ),
            "coverage_strategy": strategy,
            "coverage_model": (
                "time-limited OSCRP incumbent plus 3-5 Pareto-ranked "
                "entry/exit route candidates"
                if strategy == "exact_open_set_cover_route"
                else (
                    "many-to-many road-Pixel set cover with connected path primitives"
                    if strategy == "path_primitive_cover"
                    else "one deterministic representative observation Pixel per target"
                )
            ),
            "pareto_objectives": (
                [
                    "travel_to_entry_plus_patrol_min",
                    "patrol_duration_min",
                    "mean_expected_detection_min",
                ]
                if strategy == "exact_open_set_cover_route" else []
            ),
            "pareto_scope": (
                "rank-1 nondominated routes followed by diverse routes from "
                "later fronts when fewer than three rank-1 routes exist"
                if strategy == "exact_open_set_cover_route" else None
            ),
            "completion": "all required_patrol_groups for an Area must complete",
            "duration": "nominal travel with no hidden-fault interruption",
            "continuous_time": bool(continuous_time),
            "line_coverage": (
                "a line is covered when all currently effective line Pixels "
                "have been inspected from the mobility network"
            ),
        },
        "areas": area_rows,
        "tasks": tasks,
    }
