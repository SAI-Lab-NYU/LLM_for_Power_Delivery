from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import threading
import webbrowser
from typing import Any, Dict, Iterable, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from plotly.offline import get_plotlyjs
import uvicorn

from patrol_planning_v23 import build_patrol_macro_payload
from planning_travel_v23 import planning_travel_times_for
from simulator_core_v23 import DSRDashboardBackend
from simulator_plotly_v23 import build_grid_figure


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index_v23.html"
TEXT_INDEX_PATH = APP_DIR / "text_v23.html"
PATROL_COVERAGE_COMPARISON_PATH = (
    APP_DIR
    / "results"
    / "patrol_coverage_effectiveness_v1"
    / "patrol_coverage_comparison.html"
)

try:
    RUNTIME_DIR = APP_DIR / "runtime"
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    RUNTIME_DIR = Path(tempfile.mkdtemp(prefix="dsr_v23_runtime_"))

backend = DSRDashboardBackend(
    data_dir=APP_DIR,
    output_dir=RUNTIME_DIR,
)
backend.load_map(render_output=False)

backend_lock = threading.RLock()
server_instance: uvicorn.Server | None = None
plotly_javascript = get_plotlyjs()
controller_updates: list[Dict[str, Any]] = []
controller_next_sequence = 1
controller_generation = 0
CONTINUOUS_REPLAY_MIN_FRAME_SPACING_MIN = 0.75

CONTROLLER_REPLAY_RESULTS = {
    "parent-pf-continuous-unknown": (
        APP_DIR / "results"
        / "basic13_parent_arc_lindistflow_three_cases_v1"
        / "unknown_rolling"
    ),
    "parent-pf-continuous-rolling": (
        APP_DIR / "results"
        / "basic13_parent_arc_lindistflow_three_cases_v1"
        / "known_rolling"
    ),
    "parent-pf-continuous-one-shot": (
        APP_DIR / "results"
        / "basic13_parent_arc_lindistflow_three_cases_v1"
        / "known_one_shot"
    ),
    "greedy-2fault": APP_DIR / "results" / "greedy_debug_s1_fixed",
    "greedy": APP_DIR / "results" / "three_fault_nearest_greedy",
    "continuous-unknown-patrol": (
        APP_DIR / "results" / "continuous_event_rolling_unknown_patrol_v1"
    ),
    "continuous-rolling": (
        APP_DIR / "results" / "continuous_event_rolling_v1"
    ),
    "continuous-one-shot": (
        APP_DIR / "results" / "continuous_event_one_shot_v1"
    ),
    "one-shot": (
        APP_DIR / "results" / "gurobi_exact_no_unknown_open_loop_v1"
    ),
    "h4-rollout": APP_DIR / "results" / "gurobi_bridge_h4",
    "global": (
        APP_DIR / "results" / "gurobi_global_route_arc_v4_homogeneous_rc_unknown"
    ),
    "global-no-unknown": (
        APP_DIR / "results" / "gurobi_global_route_arc_v4_homogeneous_rc_no_unknown"
    ),
    "three-fault": (
        APP_DIR / "results" / "three_fault_information_aware_mps_lead"
    ),
}
MODEL_DOCUMENTS = {
    "rolling-horizon": (
        APP_DIR / "results" / "gurobi_bridge_h4" / "model_and_results.html"
    ),
    "global-route-arc": (
        APP_DIR
        / "results"
        / "gurobi_global_route_arc_v1"
        / "model_and_results.html"
    ),
    "rolling-horizon-en": (
        APP_DIR
        / "results"
        / "gurobi_bridge_h4"
        / "model_and_results_en.html"
    ),
    "global-route-arc-en": (
        APP_DIR
        / "results"
        / "gurobi_global_route_arc_v1"
        / "model_and_results_en.html"
    ),
}

app = FastAPI(title="DSR Grid Map v23")


def _reset_controller_updates() -> None:
    """Start a fresh web replay generation for a newly loaded scenario."""

    global controller_next_sequence, controller_generation
    controller_updates.clear()
    controller_next_sequence = 1
    controller_generation += 1


def _store_controller_update(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Append one JSON-safe controller/simulator interaction record."""

    global controller_next_sequence
    update = json.loads(json.dumps(payload))
    update["sequence"] = controller_next_sequence
    update["generation"] = controller_generation
    controller_next_sequence += 1
    controller_updates.append(update)
    if len(controller_updates) > 100:
        del controller_updates[:-100]
    return update


def _read_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Saved controller result is missing: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _saved_observation_metadata(
    observation: Dict[str, Any],
) -> Dict[str, Any]:
    faults = observation.get("faults", {})
    if isinstance(faults, list):
        faults = {fault["name"]: fault for fault in faults}
    return {
        "time_step": int(observation.get("time_step", 0)),
        "time_min": float(observation.get("time_min", 0.0)),
        "weighted_energized_kw": float(
            observation.get("weighted_energized_kw", 0.0)
        ),
        "unknown_pixel_count": int(
            observation.get("unknown_pixel_count", 0)
        ),
        "known_faults": [
            {
                "name": name,
                "analyzed": bool(fault.get("analyzed", False)),
            }
            for name, fault in faults.items()
            if bool(fault.get("active", False))
        ],
        "units": {
            name: {
                "state": unit.get("state", "Idle"),
                "position": list(unit.get("position", [0, 0])),
                "remaining_path": [
                    list(pixel)
                    for pixel in unit.get("remaining_path", [])
                ],
                "planned_destination": unit.get("planned_destination"),
                "remaining_path_min": float(
                    unit.get("remaining_path_min", 0.0)
                ),
                "analysis_remaining_min": float(
                    unit.get("analysis_remaining_min", 0.0)
                ),
                "repair_remaining_min": float(
                    unit.get("repair_remaining_min", 0.0)
                ),
            }
            for name, unit in observation.get("units", {}).items()
        },
    }


def _saved_plan_metadata(plan: Dict[str, Any]) -> Dict[str, Any]:
    sequences = plan.get("planned_task_sequences", {})
    forecast = plan.get(
        "predicted_weighted_trajectory_kw",
        plan.get("objectives", {}).get(
            "predicted_weighted_trajectory_kw", []
        ),
    )
    return {
        "selected_tasks": {
            unit: " -> ".join(sequence)
            for unit, sequence in sequences.items()
        },
        "selected_topology": plan.get("selected_topology", "hold"),
        "objectives": dict(plan.get("objectives", {})),
        "solver_status": plan.get("solver_status", "UNKNOWN"),
        "horizon_steps": plan.get(
            "horizon_steps", "global-to-restoration-end"
        ),
        "planned_task_sequences": sequences,
        "predicted_weighted_trajectory_kw": forecast,
        "area_restoration_time_min": plan.get(
            "area_restoration_time_min", {}
        ),
}


def _saved_controller_result_summary(
    replay_name: str,
    result_dir: Path,
) -> Dict[str, Any]:
    """Build compact, public result metadata for the graphical dashboard."""

    config = json.loads(
        (result_dir / "config.json").read_text(encoding="utf-8")
    )
    commands = _read_jsonl(result_dir / "commands.jsonl")
    observations = _read_jsonl(result_dir / "observations.jsonl")
    events = _read_jsonl(result_dir / "events.jsonl")
    final_state = json.loads(
        (result_dir / "final_state.json").read_text(encoding="utf-8")
    )
    run_summary_path = result_dir / "summary.json"
    run_summary = (
        json.loads(run_summary_path.read_text(encoding="utf-8"))
        if run_summary_path.exists()
        else {}
    )
    initial = observations[0] if observations else {}
    units = initial.get("units", {})
    repair_assignments: Dict[str, set[str]] = {}
    patrol_rounds: Dict[str, int] = {}
    switch_batches = []
    for index, command_record in enumerate(commands):
        command = command_record.get("command", {})
        for unit_name, action in command.get("mobile_commands", {}).items():
            if action.get("type") == "repair" and action.get("fault"):
                repair_assignments.setdefault(unit_name, set()).add(
                    str(action["fault"])
                )
            if action.get("patrol_task_id"):
                patrol_rounds[unit_name] = patrol_rounds.get(unit_name, 0) + 1
        switch_commands = dict(command.get("switch_commands", {}))
        if switch_commands:
            time_min = float(
                observations[index].get("time_min", 0.0)
                if index < len(observations) else 0.0
            )
            switch_batches.append({
                "time_min": time_min,
                "switches": switch_commands,
            })

    final_clock = final_state.get("clock", {})
    final_game = final_state.get("game", {})
    final_summary = final_state.get("summary", {})
    weighted_history = [
        float(row.get("weighted_energized_kw", 0.0))
        for row in observations
    ]
    final_weighted = float(
        final_summary.get(
            "weighted_energized_load",
            weighted_history[-1] if weighted_history else 0.0,
        )
    )
    weighted_history.append(final_weighted)
    labels = {
        "greedy-2fault": "Nearest-First Greedy: Two Faults",
        "greedy": "Nearest-First Greedy",
        "one-shot": "One-Shot Open Loop",
        "h4-rollout": "H=4 Rollout",
        "global": "Unknown",
        "global-no-unknown": "No Unknown",
        "three-fault": "Three Faults",
    }
    mps_events = [
        {
            "type": event.get("event_type"),
            "time_min": float(event.get("time_min", 0.0)),
            "position": event.get("position"),
        }
        for event in events
        if event.get("event_type") in {"MPS_CONNECTED", "MPS_DISCONNECTED"}
    ]
    return {
        "name": replay_name,
        "label": labels.get(replay_name, replay_name),
        "controller": config.get("controller", "unknown"),
        "optimization_calls": int(
            run_summary.get(
                "optimization_calls", config.get("optimization_calls", len(commands))
            )
        ),
        "rolling_reoptimization": bool(
            run_summary.get(
                "rolling_reoptimization",
                config.get("rolling_reoptimization", True),
            )
        ),
        "solver_status": run_summary.get("solver_status", "UNKNOWN"),
        "solver_mip_gap": run_summary.get("solver_mip_gap"),
        "rounds": len(commands),
        "final_time_min": float(final_clock.get("time_min", 0.0)),
        "initial_unknown_pixels": int(initial.get("unknown_pixel_count", 0)),
        "all_faults_cleared": bool(final_game.get("all_faults_cleared")),
        "all_loads_restored": bool(final_game.get("all_loads_restored")),
        "final_weighted_energized_kw": final_weighted,
        "validation_fallbacks": sum(
            str(row.get("validation", "validated")) != "validated"
            for row in commands
        ),
        "weighted_load_monotonic": all(
            later + 1e-6 >= earlier
            for earlier, later in zip(weighted_history, weighted_history[1:])
        ),
        "crew_parameters": {
            unit_name: {
                "move_speed_pixels_per_step": float(
                    unit.get("move_speed_pixels_per_step", 0.0)
                ),
                "repair_speed_resource_per_step": unit.get("repair_speed"),
            }
            for unit_name, unit in units.items()
            if unit.get("unit_type") == "RC"
        },
        "repair_assignments": {
            unit_name: sorted(faults)
            for unit_name, faults in repair_assignments.items()
        },
        "patrol_rounds": patrol_rounds,
        "switch_batches": switch_batches,
        "mps_events": mps_events,
    }


def _saved_continuous_result_summary(
    replay_name: str,
    result_dir: Path,
) -> Dict[str, Any]:
    """Build dashboard metadata for continuous-event saved results."""

    config = json.loads(
        (result_dir / "config.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (result_dir / "summary.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(
        (result_dir / "timeline.json").read_text(encoding="utf-8")
    )
    events = json.loads(
        (result_dir / "events.json").read_text(encoding="utf-8")
    )
    repair_assignments: Dict[str, set[str]] = {}
    for event in events:
        if event.get("event_type") != "REPAIR_COMPLETED":
            continue
        fault_name = str(event.get("fault", ""))
        for unit_name in event.get("details", {}).get("units", []):
            repair_assignments.setdefault(str(unit_name), set()).add(fault_name)
    switch_commands = {
        str(event["switch"]): str(event.get("details", {}).get("target_state"))
        for event in events
        if event.get("event_type") == "REMOTE_SWITCH_OPERATED"
    }
    weighted = [
        float(row.get("weighted_energized_kw", 0.0)) for row in timeline
    ]
    return {
        "name": replay_name,
        "label": "Continuous-Event One Shot",
        "controller": config["controller"],
        "optimization_calls": int(config["optimization_calls"]),
        "rolling_reoptimization": bool(config["rolling_reoptimization"]),
        "event_driven_execution": True,
        "continuous_time_milp": bool(
            config.get("continuous_time_milp", False)
        ),
        "solver_status": summary["solver_status"],
        "solver_mip_gap": summary["solver_mip_gap"],
        "rounds": len(timeline),
        "event_count": int(summary["event_count"]),
        "final_time_min": float(summary["final_time_min"]),
        "initial_unknown_pixels": 0,
        "all_faults_cleared": bool(summary["all_faults_cleared"]),
        "all_loads_restored": bool(summary["all_loads_restored"]),
        "final_weighted_energized_kw": float(
            summary["final_weighted_energized_kw"]
        ),
        "validation_fallbacks": 0,
        "weighted_load_monotonic": all(
            later + 1e-6 >= earlier
            for earlier, later in zip(weighted, weighted[1:])
        ),
        "crew_parameters": {
            "RC1": {
                "move_speed_pixels_per_step": 8.0,
                "repair_speed_resource_per_step": 5,
            },
            "RC2": {
                "move_speed_pixels_per_step": 8.0,
                "repair_speed_resource_per_step": 5,
            },
        },
        "repair_assignments": {
            name: sorted(faults)
            for name, faults in repair_assignments.items()
        },
        "patrol_rounds": {},
        "switch_batches": ([{
            "time_min": 0.0,
            "switches": switch_commands,
        }] if switch_commands else []),
        "mps_events": [],
        "actual_fault_completion_min": summary[
            "actual_fault_completion_min"
        ],
        "fixed_step_fault_completion_min": {
            "A": 25.0,
            "B": 25.0,
        },
        "maximum_fault_completion_error_min": float(
            summary["maximum_fault_completion_error_min"]
        ),
    }


def _saved_continuous_rolling_summary(
    replay_name: str,
    result_dir: Path,
) -> Dict[str, Any]:
    """Build compact metadata for continuous event-driven rolling."""

    config = json.loads(
        (result_dir / "config.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (result_dir / "summary.json").read_text(encoding="utf-8")
    )
    events = json.loads(
        (result_dir / "events.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(
        (result_dir / "timeline.json").read_text(encoding="utf-8")
    )
    assignments: Dict[str, set[str]] = {}
    for event in events:
        if event.get("event_type") != "REPAIR_COMPLETED":
            continue
        for unit in event.get("details", {}).get("units", []):
            assignments.setdefault(str(unit), set()).add(str(event["fault"]))
    remote = {
        str(event["switch"]): str(event.get("details", {}).get("target_state"))
        for event in events
        if event.get("event_type") == "REMOTE_SWITCH_OPERATED"
    }
    weighted = [float(row["weighted_energized_kw"]) for row in timeline]
    unknown_mode = replay_name == "continuous-unknown-patrol"
    return {
        "name": replay_name,
        "label": (
            "Continuous Unknown Patrol" if unknown_mode
            else "Continuous-Event Rolling"
        ),
        "controller": config["controller"],
        "optimization_calls": int(summary["optimization_calls"]),
        "rolling_reoptimization": True,
        "event_driven_execution": True,
        "continuous_time_milp": True,
        "solver_status": (
            "TIME_LIMIT_FEASIBLE"
            if "TIME_LIMIT" in summary.get("solver_statuses", [])
            else "OPTIMAL"
        ),
        "solver_mip_gap": float(summary["maximum_solver_mip_gap"]),
        "rounds": int(summary["optimization_calls"]),
        "event_count": int(summary["event_count"]),
        "final_time_min": float(summary["final_time_min"]),
        "initial_unknown_pixels": int(
            summary.get("initial_unknown_pixel_count", 0)
        ),
        "final_unknown_pixels": int(
            summary.get("final_unknown_pixel_count", 0)
        ),
        "all_faults_cleared": bool(summary["all_faults_cleared"]),
        "all_loads_restored": bool(summary["all_loads_restored"]),
        "final_weighted_energized_kw": float(
            summary["final_weighted_energized_kw"]
        ),
        "validation_fallbacks": 0,
        "weighted_load_monotonic": all(
            later + 1e-6 >= earlier
            for earlier, later in zip(weighted, weighted[1:])
        ),
        "crew_parameters": {
            name: {
                "move_speed_pixels_per_step": 8.0,
                "repair_speed_resource_per_step": 5,
            }
            for name in ("RC1", "RC2")
        },
        "repair_assignments": {
            name: sorted(faults) for name, faults in assignments.items()
        },
        "patrol_rounds": {},
        "switch_batches": ([{
            "time_min": 0.0, "switches": remote,
        }] if remote else []),
        "mps_events": [],
        "actual_fault_completion_min": summary[
            "actual_fault_completion_min"
        ],
        "continuous_one_shot_fault_completion_min": {
            "A": 23.125,
            "B": 19.666666666666664,
        },
        "fixed_step_fault_completion_min": {"A": 25.0, "B": 25.0},
        "maximum_fault_completion_error_min": float(
            summary["maximum_initial_fault_prediction_error_min"]
        ),
        "fault_discovery_min": summary.get("fault_discovery_min", {}),
    }


def _apply_continuous_snapshot(snapshot: Dict[str, Any]) -> None:
    """Apply one public continuous snapshot to the graphical backend."""

    time_min = float(snapshot["time_min"])
    backend.time_step = int((time_min + 1e-9) // backend.minutes_per_step)
    backend.current_frame = 0
    if "unknown_pixels" in snapshot:
        unknown = {
            tuple(position) for position in snapshot["unknown_pixels"]
        }
        for position, pixel in backend.graph.pixels.items():
            pixel.last_inspected_event = (
                None if position in unknown else backend.event_id
            )
    for name, target in snapshot.get("switches", {}).items():
        switch = backend.switches[name]
        if target == "closed":
            switch.close()
        else:
            switch.open()
    for name, saved in snapshot.get("faults", {}).items():
        fault = backend.faults[name]
        if saved.get("analyzed") and not fault.analyzed:
            fault.mark_analyzed("continuous-event")
        if not saved.get("active", True) and fault.active:
            fault.repair()
    for name, saved in snapshot.get("units", {}).items():
        unit = backend.mobile_units[name]
        unit.position = tuple(saved["position"])
        unit.state = str(saved["state"])
        unit.cumulative_travel_min = float(
            saved.get("cumulative_travel_min", 0.0)
        )
        unit.planned_path = [
            tuple(pixel) for pixel in saved.get("remaining_path", [])
        ]
        unit.set_patrol_context(
            saved.get("active_patrol_task_id"),
            saved.get("active_patrol_area"),
            saved.get("active_patrol_required_lines", []),
        )
    for switch in backend.switches.values():
        switch.update_knowledge(
            nodes=backend.nodes, event_id=backend.event_id
        )
    for line in backend.electrical_lines.values():
        line.update_state()
    backend.transmission_source.scenario_fault_enabled = True
    backend.transmission_source.update_availability(backend.time_step)
    backend._refresh_derived_states()


def _continuous_travel_legs(
    timeline: list[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    """Recover exact road-Pixel crossing times from departure events."""

    legs: list[Dict[str, Any]] = []
    for snapshot in timeline:
        for event in snapshot.get("events", []):
            if event.get("event_type") != "MOBILE_DEPARTED":
                continue
            unit_name = str(event["unit"])
            unit = backend.mobile_units[unit_name]
            start = tuple(event["position"])
            path = [tuple(point) for point in event["details"].get("path", [])]
            if not path:
                continue
            duration = float(event["details"]["travel_time_min"])
            raw_segment_times = [
                unit.movement_cost(first, second) * backend.minutes_per_step
                for first, second in zip([start, *path], path)
            ]
            raw_total = sum(raw_segment_times)
            scale = duration / raw_total if raw_total > 0.0 else 0.0
            elapsed = 0.0
            crossings = []
            for position, segment_time in zip(path, raw_segment_times):
                elapsed += segment_time * scale
                crossings.append((elapsed, position))
            start_time = float(event["time_min"])
            legs.append({
                "unit": unit_name,
                "start_time": start_time,
                "end_time": start_time + duration,
                "start_position": start,
                "path": path,
                "crossings": crossings,
                "start_cumulative_travel_min": float(
                    snapshot["units"][unit_name].get(
                        "cumulative_travel_min", 0.0
                    )
                ),
            })
    return legs


def _continuous_replay_frame_times(
    travel_legs: list[Dict[str, Any]],
    start_time: float,
    end_time: float,
    *,
    include_start: bool = False,
) -> list[float]:
    """Return merged Pixel crossings plus the committed event time."""

    frame_times = {round(end_time, 6)}
    if include_start:
        frame_times.add(round(start_time, 6))
    for leg in travel_legs:
        departure = float(leg["start_time"])
        for elapsed, _position in leg["crossings"]:
            crossing_time = departure + float(elapsed)
            if start_time + 1e-6 < crossing_time < end_time - 1e-6:
                frame_times.add(round(crossing_time, 6))
    ordered = sorted(frame_times)
    if len(ordered) <= 1:
        return ordered
    selected = [ordered[0]]
    for frame_time in ordered[1:-1]:
        if (
            frame_time - selected[-1]
            >= CONTINUOUS_REPLAY_MIN_FRAME_SPACING_MIN - 1e-6
        ):
            selected.append(frame_time)
    if ordered[-1] != selected[-1]:
        selected.append(ordered[-1])
    return selected


def _apply_continuous_travel_positions(
    travel_legs: list[Dict[str, Any]],
    time_min: float,
) -> None:
    """Place every in-transit unit at its latest crossed road Pixel."""

    for leg in travel_legs:
        start_time = float(leg["start_time"])
        end_time = float(leg["end_time"])
        if time_min < start_time - 1e-6 or time_min >= end_time - 1e-6:
            continue
        elapsed = max(0.0, time_min - start_time)
        crossed_count = 0
        position = tuple(leg["start_position"])
        for crossing_elapsed, crossing_position in leg["crossings"]:
            if crossing_elapsed > elapsed + 1e-6:
                break
            crossed_count += 1
            position = tuple(crossing_position)
        unit = backend.mobile_units[str(leg["unit"])]
        if crossed_count:
            previous_position = (
                tuple(leg["start_position"])
                if crossed_count == 1
                else tuple(leg["path"][crossed_count - 2])
            )
            unit.heading = (
                position[0] - previous_position[0],
                position[1] - previous_position[1],
            )
        unit.position = position
        unit.state = "Moving"
        unit.planned_path = list(leg["path"][crossed_count:])
        unit.cumulative_travel_min = (
            float(leg["start_cumulative_travel_min"]) + elapsed
        )


def _apply_continuous_inspection_progress(
    travel_legs: list[Dict[str, Any]],
    start_time: float,
    frame_time: float,
) -> None:
    """Reveal Pixels inspected while mobile units cross an unknown region."""

    if frame_time <= start_time + 1e-6:
        return
    for leg in travel_legs:
        unit = backend.mobile_units[str(leg["unit"])]
        departure = float(leg["start_time"])
        for elapsed, crossing_position in leg["crossings"]:
            crossing_time = departure + float(elapsed)
            if not (start_time + 1e-6 < crossing_time <= frame_time + 1e-6):
                continue
            for position, pixel in backend.graph.pixels.items():
                if max(
                    abs(position[0] - crossing_position[0]),
                    abs(position[1] - crossing_position[1]),
                ) <= unit.inspection_range:
                    pixel.last_inspected_event = backend.event_id
    for switch in backend.switches.values():
        switch.update_knowledge(nodes=backend.nodes, event_id=backend.event_id)
    for line in backend.electrical_lines.values():
        line.update_state()


def _continuous_animation_frames(
    previous_snapshot: Dict[str, Any],
    current_snapshot: Dict[str, Any],
    travel_legs: list[Dict[str, Any]],
    *,
    include_start: bool = False,
) -> list[Dict[str, Any]]:
    """Render exact road-Pixel motion between two committed event states."""

    previous_time = float(previous_snapshot["time_min"])
    current_time = float(current_snapshot["time_min"])
    frame_times = _continuous_replay_frame_times(
        travel_legs,
        previous_time,
        current_time,
        include_start=include_start,
    )
    frames = []
    for frame_number, frame_time in enumerate(frame_times, start=1):
        is_event_frame = abs(frame_time - current_time) <= 1e-6
        base_snapshot = current_snapshot if is_event_frame else previous_snapshot
        _apply_continuous_snapshot(base_snapshot)
        if not is_event_frame and "unknown_pixels" in previous_snapshot:
            _apply_continuous_inspection_progress(
                travel_legs, previous_time, frame_time
            )
        _apply_continuous_travel_positions(travel_legs, frame_time)
        frame = {
            "frame": frame_number,
            "time_min": round(frame_time, 6),
            "kind": "event" if is_event_frame else "movement",
        }
        frame["mobile_units"] = {
            name: {
                "position": list(unit.position),
                "heading": list(unit.heading),
                "unit_type": unit.unit_type,
                "inspection_range": unit.inspection_range,
            }
            for name, unit in backend.mobile_units.items()
        }
        if "unknown_pixels" in base_snapshot:
            graph = backend.graph
            frame["unknown_matrix"] = [
                [
                    0 if graph.get_pixel((x, y)).is_inspected(
                        backend.event_id
                    ) else 1
                    for x in range(graph.x_min, graph.x_max + 1)
                ]
                for y in range(graph.y_min, graph.y_max + 1)
            ]
        frames.append(frame)
    return frames


def _load_saved_continuous_replay(result_dir: Path) -> Dict[str, Any]:
    """Expose exact-time event snapshots through the existing replay stream."""

    plan = json.loads(
        (result_dir / "global_plan.json").read_text(encoding="utf-8")
    )
    timeline = json.loads(
        (result_dir / "timeline.json").read_text(encoding="utf-8")
    )
    if not timeline:
        raise ValueError("Saved continuous replay has no timeline snapshots.")
    _reset_controller_updates()
    backend.load_map(render_output=False)
    backend.load_faults_and_unknown(
        render_output=False,
        unknown_regions=[],
    )
    initial_state = _state_payload(
        "Continuous-event replay loaded before the fixed plan starts."
    )
    plan_metadata = _saved_plan_metadata(plan)
    travel_legs = _continuous_travel_legs(timeline)
    for index, snapshot in enumerate(timeline, start=1):
        previous_snapshot = timeline[index - 2] if index > 1 else snapshot
        current_time = float(snapshot["time_min"])
        animation_frames = _continuous_animation_frames(
            previous_snapshot,
            snapshot,
            travel_legs,
            include_start=index == 1,
        )

        _apply_continuous_snapshot(snapshot)
        _apply_continuous_travel_positions(travel_legs, current_time)
        graphical_state = _state_payload(
            f"Continuous event {index} committed at "
            f"{current_time:g} min."
        )
        graphical_state["clock"]["time_min"] = current_time
        graphical_state["summary"]["time_min"] = current_time
        graphical_state["summary"]["continuous_event_index"] = index
        graphical_state["summary"]["weighted_energized_load"] = float(
            snapshot["weighted_energized_kw"]
        )
        visual_units = {
            item["name"]: {
                "state": item["state"],
                "position": item["position"],
                "cumulative_travel_min": item["cumulative_travel_min"],
                "remaining_path": item["remaining_path"],
                "remaining_path_min": item["remaining_path_min"],
                "planned_destination": item["planned_destination"],
            }
            for item in graphical_state["commands"]["mobile_units"]
        }
        current_plan = copy.deepcopy(plan_metadata)
        current_plan["objectives"][
            "predicted_next_weighted_energized_kw"
        ] = float(snapshot["weighted_energized_kw"])
        current_plan["continuous_time"] = True
        _store_controller_update({
            "round": index,
            "observation": {
                "time_step": index - 1,
                "time_min": current_time,
                "weighted_energized_kw": float(
                    snapshot["weighted_energized_kw"]
                ),
                "unknown_pixel_count": 0,
                "known_faults": [
                    {"name": name, "analyzed": bool(item["analyzed"])}
                    for name, item in snapshot["faults"].items()
                    if item["active"]
                ],
                "units": visual_units,
            },
            "plan": current_plan,
            "validation": "continuous-event exact",
            "command": {"mobile_commands": {}, "switch_commands": {}},
            "animation_frames": animation_frames,
            "events": snapshot.get("events", []),
            "state": graphical_state,
        })
    return {
        "ok": True,
        "message": (
            f"Loaded {len(timeline)} exact-time event states with "
            "road-pixel movement frames for replay."
        ),
        "generation": controller_generation,
        "round_count": len(controller_updates),
        "initial_state": initial_state,
    }


def _load_saved_continuous_rolling_replay(
    result_dir: Path,
) -> Dict[str, Any]:
    """Rebuild graphical states for exact-time rolling decisions."""

    config = json.loads(
        (result_dir / "config.json").read_text(encoding="utf-8")
    )
    decisions = _read_jsonl(result_dir / "decisions.jsonl")
    observations = _read_jsonl(result_dir / "observations.jsonl")
    plans = _read_jsonl(result_dir / "plans.jsonl")
    timeline = json.loads(
        (result_dir / "timeline.json").read_text(encoding="utf-8")
    )
    if not (len(decisions) == len(observations) == len(plans)):
        raise ValueError(
            "Continuous rolling replay requires one observation and plan "
            "per decision."
        )
    _reset_controller_updates()
    backend.load_map(render_output=False)
    backend.load_faults_and_unknown(
        render_output=False,
        unknown_regions=(
            None
            if config.get("unknown_regions") == "scenario-default"
            else config.get("unknown_regions", [])
        ),
    )
    initial_state = _state_payload(
        "Continuous rolling replay loaded before decision 1."
    )

    travel_legs = _continuous_travel_legs(timeline)
    used_timeline_indices: set[int] = set()
    previous_replay_snapshot: Dict[str, Any] | None = None
    for index, (decision, observation, plan) in enumerate(
        zip(decisions, observations, plans), start=1
    ):
        decision_time = float(decision["time_min"])
        decision_index = int(decision["decision"]) - 1
        candidates = [
            (row_index, row)
            for row_index, row in enumerate(timeline)
            if int(row.get("decision_index", -1)) == decision_index
            and abs(float(row["time_min"]) - decision_time) <= 1e-6
        ]
        if not candidates:
            raise ValueError(
                f"Missing rolling replay state for decision {index}."
            )
        state_index, snapshot = candidates[-1]
        used_timeline_indices.add(state_index)
        animation_frames = _continuous_animation_frames(
            previous_replay_snapshot or snapshot,
            snapshot,
            travel_legs,
            include_start=previous_replay_snapshot is None,
        )
        _apply_continuous_snapshot(snapshot)
        _apply_continuous_travel_positions(travel_legs, decision_time)
        graphical_state = _state_payload(
            f"Continuous rolling decision {index} dispatched at "
            f"{decision_time:g} min."
        )
        graphical_state["clock"]["time_min"] = decision_time
        graphical_state["summary"]["time_min"] = decision_time
        graphical_state["summary"]["continuous_event_index"] = index
        graphical_state["summary"]["weighted_energized_load"] = float(
            snapshot["weighted_energized_kw"]
        )
        plan_metadata = _saved_plan_metadata(plan)
        plan_metadata["objectives"][
            "predicted_next_weighted_energized_kw"
        ] = float(snapshot["weighted_energized_kw"])
        plan_metadata["continuous_time"] = True
        _store_controller_update({
            "round": index,
            "observation": _saved_observation_metadata(observation),
            "plan": plan_metadata,
            "validation": "continuous rolling exact",
            "command": decision["command"],
            "animation_frames": animation_frames,
            "events": [
                *decision.get("trigger_events", []),
                *decision.get("dispatched_events", []),
            ],
            "state": graphical_state,
        })
        previous_replay_snapshot = snapshot

    final_index = len(timeline) - 1
    final_snapshot = timeline[final_index]
    if final_index not in used_timeline_indices:
        animation_frames = _continuous_animation_frames(
            previous_replay_snapshot or final_snapshot,
            final_snapshot,
            travel_legs,
            include_start=previous_replay_snapshot is None,
        )
        _apply_continuous_snapshot(final_snapshot)
        final_time = float(final_snapshot["time_min"])
        _apply_continuous_travel_positions(travel_legs, final_time)
        graphical_state = _state_payload(
            f"Continuous rolling restoration completed at {final_time:g} min."
        )
        graphical_state["clock"]["time_min"] = final_time
        graphical_state["summary"]["time_min"] = final_time
        graphical_state["summary"]["continuous_event_index"] = len(decisions) + 1
        graphical_state["summary"]["weighted_energized_load"] = float(
            final_snapshot["weighted_energized_kw"]
        )
        plan_metadata = _saved_plan_metadata(plans[-1])
        plan_metadata["objectives"][
            "predicted_next_weighted_energized_kw"
        ] = float(final_snapshot["weighted_energized_kw"])
        visual_units = {
            item["name"]: {
                "state": item["state"],
                "position": item["position"],
                "cumulative_travel_min": item["cumulative_travel_min"],
                "remaining_path": item["remaining_path"],
                "remaining_path_min": item["remaining_path_min"],
                "planned_destination": item["planned_destination"],
            }
            for item in graphical_state["commands"]["mobile_units"]
        }
        _store_controller_update({
            "round": len(decisions) + 1,
            "observation": {
                "time_step": len(decisions),
                "time_min": final_time,
                "weighted_energized_kw": float(
                    final_snapshot["weighted_energized_kw"]
                ),
                "unknown_pixel_count": 0,
                "known_faults": [],
                "units": visual_units,
            },
            "plan": plan_metadata,
            "validation": "continuous rolling final event",
            "command": {"mobile_commands": {}, "switch_commands": {}},
            "animation_frames": animation_frames,
            "events": final_snapshot.get("events", []),
            "state": graphical_state,
        })
    return {
        "ok": True,
        "message": (
            f"Loaded {len(controller_updates)} continuous rolling states "
            "with road-pixel movement frames."
        ),
        "generation": controller_generation,
        "round_count": len(controller_updates),
        "initial_state": initial_state,
    }


def _apply_saved_unit_parameters(observation: Dict[str, Any]) -> None:
    """Restore historical runtime parameters before replaying old commands."""

    for unit_name, saved in observation.get("units", {}).items():
        unit = backend.mobile_units.get(unit_name)
        if unit is None:
            continue
        move_speed = saved.get("move_speed_pixels_per_step")
        if move_speed is not None:
            unit.move_speed = int(move_speed)
        repair_speed = saved.get("repair_speed")
        if repair_speed is not None and unit_name in backend.repair_crews:
            unit.repair_speed = int(repair_speed)


def _load_saved_controller_replay(result_dir: Path) -> Dict[str, Any]:
    """Deterministically rebuild graphical replay rounds from saved commands."""

    config_path = result_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Saved controller configuration is missing: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    commands = _read_jsonl(result_dir / "commands.jsonl")
    plans_path = result_dir / "plans.jsonl"
    if plans_path.exists():
        plans = _read_jsonl(plans_path)
    else:
        global_plan_path = result_dir / "global_plan.json"
        if not global_plan_path.exists():
            raise FileNotFoundError(
                "Saved controller result requires plans.jsonl or "
                f"global_plan.json: {result_dir}"
            )
        global_plan = json.loads(global_plan_path.read_text(encoding="utf-8"))
        plans = [global_plan for _ in commands]
    observations = _read_jsonl(result_dir / "observations.jsonl")
    if not commands:
        raise ValueError(f"Saved controller result has no commands: {result_dir}")
    if not (len(commands) == len(plans) == len(observations)):
        raise ValueError(
            "Saved controller replay requires one observation, plan, and "
            "command record per round."
        )

    _reset_controller_updates()
    backend.load_map(render_output=False)
    backend.configure_max_time_step(int(config.get("max_steps", 20)))
    configured_unknown_regions = config.get(
        "unknown_regions", "scenario-default"
    )
    load_options: Dict[str, Any] = {"render_output": False}
    if configured_unknown_regions != "scenario-default":
        load_options["unknown_regions"] = configured_unknown_regions
    if config.get("fault_scenario") is not None:
        load_options["fault_scenario"] = config["fault_scenario"]
    if config.get("unit_initial_positions") is not None:
        load_options["unit_initial_positions"] = config[
            "unit_initial_positions"
        ]
    backend.load_faults_and_unknown(**load_options)
    _apply_saved_unit_parameters(observations[0])
    initial_state = _state_payload(
        "Saved global controller replay loaded at the initial state."
    )

    for index, (command_record, plan_record, observation_record) in enumerate(
        zip(commands, plans, observations), start=1
    ):
        command = command_record.get("command", {})
        mobile_commands, switch_operations = _parse_command_payload(command)
        result = backend.execute_current_step(
            mobile_commands=mobile_commands,
            switch_operations=switch_operations,
            frames_per_step=backend.physics_frames_per_step,
            collect_frame_snapshots=True,
        )
        frame_snapshots = result.pop("frame_snapshots")
        animation_frames = [
            {
                "frame": frame_index,
                "figure": json.loads(build_grid_figure(snapshot).to_json()),
            }
            for frame_index, snapshot in enumerate(
                frame_snapshots, start=1
            )
        ]
        graphical_state = _state_payload(_command_result_message(result))
        plan_metadata = _saved_plan_metadata(plan_record)
        plan_metadata["objectives"].setdefault(
            "predicted_next_weighted_energized_kw",
            graphical_state["summary"]["weighted_energized_load"],
        )
        _store_controller_update({
            "round": int(command_record.get("round", index)),
            "observation": _saved_observation_metadata(observation_record),
            "plan": plan_metadata,
            "validation": str(
                command_record.get("validation", "validated")
            ),
            "command": {
                "mobile_commands": mobile_commands,
                "switch_commands": {
                    name: state for name, state in switch_operations
                },
            },
            "animation_frames": animation_frames,
            "events": result.get("events", []),
            "state": graphical_state,
        })

    return {
        "ok": True,
        "message": (
            f"Loaded {len(controller_updates)} saved global controller "
            "rounds for simulator replay."
        ),
        "generation": controller_generation,
        "round_count": len(controller_updates),
        "initial_state": initial_state,
    }


TABLE_DEFINITIONS = {
    "sources": {
        "title": "Power Sources",
        "headers": [
            "Source", "Node", "Status", "Restore Step", "Restore Time/min", "P/kW",
            "Q/kVar", "Isolated Switches",
        ],
    },
    "mps": {
        "title": "MPS States",
        "headers": [
            "MPS", "State", "Position", "Energy/kWh", "P Limit",
            "S Limit", "P/kW", "Q/kVar", "Connected Areas",
        ],
    },
    "power_clusters": {
        "title": "Power Clusters",
        "headers": [
            "Areas", "Nodes", "Load P", "Load Q", "MPS P", "MPS Q",
            "Grid P", "Grid Q", "Status",
        ],
    },
    "energized_loads": {
        "title": "Energized Loads",
        "headers": ["Node", "P/kW", "Q/kVar", "Weight"],
    },
    "faults": {
        "title": "Known Faults",
        "headers": [
            "Fault", "Position", "Knowledge", "Status", "Required Resource",
            "Remaining Resource", "Area",
        ],
    },
    "areas": {
        "title": "Static Areas",
        "headers": [
            "Area", "Nodes", "Lines", "Boundary Switches", "Knowledge",
            "Energizable",
        ],
    },
    "switches": {
        "title": "Switch States",
        "headers": [
            "Switch", "Node", "Line", "State", "Node Energized",
            "Node-side Area", "Line-side Area", "Operation", "Time/min",
        ],
    },
    "elements": {
        "title": "Element States",
        "headers": ["Type", "Name", "Knowledge", "Power", "State", "Area"],
    },
}


COMMAND_SCHEMA = {
    "switch_commands": {
        "S1": "opened",
        "S2": "closed",
    },
    "mobile_commands": {
        "RC1": {"type": "continue"},
        "RC2": {"type": "move", "waypoints": [[0, 0]]},
        "RC_manual_example": {
            "type": "operate_switch",
            "switch": "S8",
            "target_state": "closed",
        },
        "Scott1": {"type": "stay"},
        "MPS1": {"type": "supply", "p": 800, "q": 440},
    },
}


def _summary_payload_for(source: DSRDashboardBackend) -> Dict[str, Any]:
    known_faults = sum(
        fault.active and fault.is_known(source.event_id)
        for fault in source.faults.values()
    )

    energized_active_power = source.restored_active_power()
    weighted_energized_load = sum(
        node.active_power * node.load_weight
        for node in source.nodes.values()
        if node.has_load and node.energized
    )

    return {
        "stage": source.stage,
        "time_step": source.time_step,
        "time_min": source.time_step * source.minutes_per_step,
        "event_id": source.event_id,
        "known_faults": known_faults,
        "energized_active_power": source._format_load_value(
            energized_active_power
        ),
        "weighted_energized_load": source._format_load_value(
            weighted_energized_load
        ),
        "source_status": (
            "Available"
            if source.transmission_source is not None
            and source.transmission_source.available
            else "Outage"
        ),
        "source_restore_step": (
            source.transmission_source.restoration_time_step
            if source.transmission_source is not None
            else 0
        ),
        "source_restore_time_min": (
            source.transmission_source.restoration_time_step
            * source.minutes_per_step
            if source.transmission_source is not None
            else 0.0
        ),
        "grid_active_power": source._format_load_value(
            source.transmission_source.active_power
            if source.transmission_source is not None
            else 0.0
        ),
        "grid_reactive_power": source._format_load_value(
            source.transmission_source.reactive_power
            if source.transmission_source is not None
            else 0.0
        ),
    }


def _table_payload_for(
    source: DSRDashboardBackend,
) -> Dict[str, Dict[str, Any]]:
    rows_by_key = {
        "sources": source.source_rows(),
        "mps": source.mps_rows(),
        "power_clusters": source.power_cluster_rows_cache,
        "energized_loads": source.energized_load_rows(),
        "faults": source.fault_rows(known_only=True),
        "areas": source.area_rows(),
        "switches": source.switch_rows(),
        "elements": source.element_rows(),
    }
    return {
        key: {
            **definition,
            "rows": rows_by_key[key],
        }
        for key, definition in TABLE_DEFINITIONS.items()
    }


def _mobile_command_metadata_for(
    source: DSRDashboardBackend,
) -> list[Dict[str, Any]]:
    metadata = []
    for unit in source.mobile_units.values():
        repairable_faults = source.repairable_fault_names(unit.name)
        item: Dict[str, Any] = {
            "name": unit.name,
            "type": unit.unit_type,
            "state": unit.state,
            "position": list(unit.position),
            "inspection_range": unit.inspection_range,
            "move_speed": unit.move_speed,
            "move_speed_pixels_per_step": unit.move_speed,
            "move_speed_pixels_per_min": (
                unit.move_speed / source.minutes_per_step
            ),
            "repair_speed": (
                unit.repair_speed if unit.unit_type == "RC" else None
            ),
            "remaining_resources": (
                unit.remaining_resources if unit.unit_type == "RC" else None
            ),
            "total_resources": (
                unit.total_resources if unit.unit_type == "RC" else None
            ),
            "remaining_path_pixels": len(unit.planned_path),
            "remaining_path": [list(point) for point in unit.planned_path],
            "planned_destination": (
                list(unit.planned_path[-1]) if unit.planned_path else None
            ),
            "remaining_path_min": round(sum(
                unit.movement_cost(first, second) * source.minutes_per_step
                for first, second in zip(
                    [unit.position, *unit.planned_path],
                    unit.planned_path,
                )
            ), 6),
            "cumulative_travel_min": round(
                float(getattr(unit, "cumulative_travel_min", 0.0)), 6
            ),
            "cumulative_travel_pixels": int(getattr(
                unit, "cumulative_travel_pixels", 0
            )),
            "repair_target": unit.repair_fault_name,
            "fault_analysis_enabled": unit.fault_analysis_enabled,
            "analysis_locked": unit.analysis_locked,
            "analysis_faults": list(unit.analysis_fault_names),
            "operating_switch": getattr(
                unit, "switch_operation_name", None
            ),
            "switch_target_state": getattr(
                unit, "switch_target_state", None
            ),
            "switch_operation_remaining_min": max(
                0.0,
                getattr(unit, "switch_operation_time_min", 0.0)
                - getattr(unit, "switch_operation_elapsed_min", 0.0),
            ),
            "active_patrol_task_id": getattr(
                unit, "active_patrol_task_id", None
            ),
            "active_patrol_area": getattr(
                unit, "active_patrol_area", None
            ),
            "active_patrol_required_lines": list(getattr(
                unit, "active_patrol_required_lines", ()
            )),
            "repairable_faults": [
                {
                    "name": fault_name,
                    "maximum_amount": source.maximum_repair_amount(
                        unit.name, fault_name
                    ),
                }
                for fault_name in repairable_faults
            ],
        }

        if unit.unit_type == "MPS":
            context = source.mps_supply_context(unit.name)
            item.update({
                "energy_kwh": unit.energy,
                "energy_capacity_kwh": unit.energy_capacity,
                "energy": unit.energy,
                "energy_capacity": unit.energy_capacity,
                "p_limit": unit.p_limit,
                "s_limit": unit.s_limit,
                "current_p": unit.current_p,
                "current_q": unit.current_q,
                "supply_context": context,
                "current_command": (
                    f"Supply P={unit.current_p:g}, Q={unit.current_q:g}"
                    if unit.state == "Supplying"
                    else (
                        f"Move: {len(unit.planned_path)} Pixels remaining"
                        if unit.state == "Moving"
                        else "Idle"
                    )
                ),
            })
        else:
            item["current_command"] = (
                f"Operate {unit.switch_operation_name} to "
                f"{unit.switch_target_state}"
                if unit.state == "OperatingSwitch"
                else
                "Automatic analysis: " + ", ".join(unit.analysis_fault_names)
                if unit.analysis_locked
                else (
                    f"Move: {len(unit.planned_path)} Pixels remaining"
                    if unit.state == "Moving"
                    else (
                        f"Repaire Fault {unit.repair_fault_name}: "
                        f"{unit.pending_repair_amount}/step"
                        if unit.state == "Repairing"
                        else "Idle"
                    )
                )
            )
        metadata.append(item)
    return metadata


def _switch_command_metadata_for(
    source: DSRDashboardBackend,
) -> list[Dict[str, Any]]:
    active_operators = {
        crew.switch_operation_name: crew.name
        for crew in source.repair_crews.values()
        if crew.switch_operation_name is not None
    }
    return [
        {
            "name": switch.name,
            "state": switch.state,
            "node": switch.node_name,
            "terminal_position": list(
                source.nodes[switch.node_name].position
            ),
            "line": switch.line_name,
            "operation_type": switch.operation_type,
            "operation_time_min": switch.operation_time_min,
            "operated_by": active_operators.get(switch.name),
            "disabled": (
                switch.state == "unknown"
                or switch.operation_type == "manual"
            ),
            "remote_commandable": (
                switch.state != "unknown"
                and switch.operation_type == "remote"
            ),
        }
        for switch in source.switches.values()
    ]


def _pixel_payload_for(
    source: DSRDashboardBackend,
) -> Dict[str, Dict[str, Any]]:
    if source.graph is None:
        return {}
    return {
        f"{x},{y}": {
            "position": [x, y],
            "knowledge": (
                "Known"
                if pixel.is_inspected(source.event_id)
                else "Unknown"
            ),
            "road_level": pixel.road_level,
            "nodes": sorted(pixel.node_names),
            "lines": sorted(pixel.line_names),
            "faults": sorted(
                fault_name
                for fault_name in pixel.active_fault_ids
                if (
                    pixel.is_inspected(source.event_id)
                    and fault_name in source.faults
                    and source.faults[fault_name].is_known(source.event_id)
                )
            ),
        }
        for (x, y), pixel in source.graph.pixels.items()
    }


def _scenario_payload_for(source: DSRDashboardBackend) -> Dict[str, Any]:
    return {
        "unknown_regions": source.unknown_region_payload(),
        "can_edit": not source.game_score.started,
    }


def _inspection_payload_for(source: DSRDashboardBackend) -> Dict[str, Any]:
    """Return compact unknown-frontier targets without exposing hidden faults."""

    if source.graph is None:
        return {"unknown_pixel_count": 0, "frontier_pixels": []}
    unknown = {
        position
        for position, pixel in source.graph.pixels.items()
        if not pixel.is_inspected(source.event_id)
    }
    frontier = []
    for x, y in sorted(unknown):
        pixel = source.graph.pixels[(x, y)]
        operational = bool(
            pixel.node_names or pixel.line_names or pixel.road_names
        )
        if not operational:
            continue
        neighbors = (
            (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)
        )
        if any(
            neighbor in source.graph.pixels and neighbor not in unknown
            for neighbor in neighbors
        ):
            frontier.append([x, y])
    return {
        "unknown_pixel_count": len(unknown),
        "frontier_pixels": frontier,
    }


_ROLLOUT_LOCKED_STATES = {
    "Moving",
    "Analyzing",
    "Repairing",
    "OperatingSwitch",
}


def _rollout_weighted_load(source: DSRDashboardBackend) -> float:
    return float(sum(
        node.active_power * node.load_weight
        for node in source.nodes.values()
        if node.has_load and node.energized
    ))


def _rollout_remote_recourse(
    source: DSRDashboardBackend,
) -> list[tuple[str, str]]:
    """Choose one legal future remote action by restored load and takeover."""

    current_load = _rollout_weighted_load(source)
    current_takeovers = sum(
        source.mps_supply_context(name).get("grid_connected", False)
        for name in source.mps_units
        if source.mps_units[name].state == "Supplying"
    )
    best_score = (current_load, int(current_takeovers), 0)
    best_operation: list[tuple[str, str]] = []
    for switch in source.switches.values():
        if switch.state == "unknown" or switch.operation_type != "remote":
            continue
        target = "closed" if switch.state == "opened" else "opened"
        try:
            candidate, _ = source.validate_switch_plan([
                [switch.name, target]
            ])
        except Exception:
            continue
        candidate_load = _rollout_weighted_load(candidate)
        candidate_takeovers = sum(
            candidate.mps_supply_context(name).get("grid_connected", False)
            for name in candidate.mps_units
            if candidate.mps_units[name].state == "Supplying"
        )
        score = (candidate_load, int(candidate_takeovers), -1)
        if score > best_score:
            best_score = score
            best_operation = [(switch.name, target)]
    return best_operation


def _rollout_task_complete(
    source: DSRDashboardBackend,
    unit_name: str,
    task: Dict[str, Any],
) -> bool:
    unit = source.mobile_units[unit_name]
    task_type = str(task.get("type", "stay")).lower()
    if task_type == "continue_current":
        return unit.state not in _ROLLOUT_LOCKED_STATES
    if task_type == "fault":
        fault = source.faults[str(task["fault"])]
        return (
            not fault.active
            or (fault.analyzed and not bool(task.get("allow_repair")))
        )
    if task_type == "manual_switch":
        switch = source.switches[str(task["switch"])]
        return switch.state == str(task["target_state"])
    if task_type == "frontier":
        return (
            tuple(unit.position) == tuple(task["position"])
            and unit.state != "Moving"
        )
    return False


def _rollout_selected_command(
    source: DSRDashboardBackend,
    unit_name: str,
    tasks: list[Dict[str, Any]],
    task_index: int,
) -> tuple[Dict[str, Any], int]:
    unit = source.mobile_units[unit_name]
    while (
        task_index < len(tasks)
        and _rollout_task_complete(source, unit_name, tasks[task_index])
    ):
        task_index += 1

    if unit.state in _ROLLOUT_LOCKED_STATES:
        return {"type": "continue"}, task_index
    if task_index >= len(tasks):
        if unit.unit_type == "MPS" and unit.state == "Supplying":
            return {"type": "continue"}, task_index
        return {"type": "stay"}, task_index

    task = tasks[task_index]
    task_type = str(task.get("type", "stay")).lower()
    if task_type == "continue_current":
        return {"type": "continue"}, task_index
    if task_type == "fault":
        fault = source.faults[str(task["fault"])]
        if tuple(unit.position) != tuple(fault.position):
            return {
                "type": "move",
                "waypoints": [list(fault.position)],
            }, task_index
        if not fault.analyzed:
            return {"type": "stay"}, task_index
        if not bool(task.get("allow_repair")):
            return {"type": "stay"}, task_index
        amount = source.maximum_repair_amount(unit_name, fault.name)
        if amount <= 0:
            return {"type": "stay"}, task_index
        return {
            "type": "repair",
            "fault": fault.name,
            "amount": amount,
        }, task_index
    if task_type == "manual_switch":
        switch = source.switches[str(task["switch"])]
        terminal = source.nodes[switch.node_name].position
        if tuple(unit.position) != tuple(terminal):
            return {
                "type": "move",
                "waypoints": [list(terminal)],
            }, task_index
        return {
            "type": "operate_switch",
            "switch": switch.name,
            "target_state": str(task["target_state"]),
        }, task_index
    if task_type == "frontier":
        return {
            "type": "move",
            "waypoints": [list(task["position"])],
        }, task_index
    if task_type == "deploy_mps":
        target_node = source.nodes[str(task["node"])]
        if tuple(unit.position) != tuple(target_node.position):
            return {
                "type": "move",
                "waypoints": [list(target_node.position)],
            }, task_index
        context = source.mps_supply_context(unit_name)
        if (
            context.get("available")
            and not context.get("grid_connected")
            and float(context.get("load_p", 0.0)) > 0
        ):
            return {
                "type": "supply",
                "p": float(context["load_p"]),
                "q": float(context["load_q"]),
            }, task_index
        return {"type": "stay"}, task_index
    if task_type == "hold_supply":
        context = source.mps_supply_context(unit_name)
        return {
            "type": (
                "stay" if context.get("grid_connected") else "continue"
            )
        }, task_index
    return {"type": "stay"}, task_index


def _normalize_rollout_tasks(
    source: DSRDashboardBackend,
    unit_name: str | None,
    raw_tasks: Iterable[Dict[str, Any]],
) -> list[Dict[str, Any]]:
    if unit_name is None:
        if list(raw_tasks):
            raise ValueError("A topology-only rollout cannot contain mobile tasks.")
        return []
    if unit_name not in source.mobile_units:
        raise KeyError(f"Unknown mobile unit: {unit_name}")

    public_fault_names = {
        fault.name
        for fault in source.faults.values()
        if fault.is_known(source.event_id)
    }
    public_frontiers = {
        tuple(point)
        for point in _inspection_payload_for(source)["frontier_pixels"]
    }
    public_node_names = {
        node.name
        for node in source.nodes.values()
        if node.knowledge_state(source.event_id) == "known"
    }
    normalized = []
    for raw in raw_tasks:
        task = dict(raw)
        task_type = str(task.get("type", "stay")).lower()
        if task_type == "fault":
            fault_name = str(task.get("fault", ""))
            if fault_name not in public_fault_names:
                raise ValueError(f"Fault {fault_name} is not publicly known.")
            task["fault"] = fault_name
            task["allow_repair"] = bool(
                source.faults[fault_name].analyzed
                and task.get("allow_repair")
            )
        elif task_type == "manual_switch":
            switch_name = str(task.get("switch", ""))
            if switch_name not in source.switches:
                raise KeyError(f"Unknown switch: {switch_name}")
            switch = source.switches[switch_name]
            if switch.state == "unknown" or switch.operation_type != "manual":
                raise ValueError(
                    f"Switch {switch_name} is not a known manual switch."
                )
            target_state = str(task.get("target_state", "")).lower()
            if target_state not in {"opened", "closed"}:
                raise ValueError("Switch target must be opened or closed.")
            task.update({"switch": switch_name, "target_state": target_state})
        elif task_type == "frontier":
            position = tuple(int(value) for value in task.get("position", []))
            if position not in public_frontiers:
                raise ValueError(f"Pixel {position} is not a public frontier.")
            task["position"] = list(position)
        elif task_type == "deploy_mps":
            node_name = str(task.get("node", ""))
            if node_name not in public_node_names:
                raise ValueError(f"Node {node_name} is not publicly known.")
            if source.mobile_units[unit_name].unit_type != "MPS":
                raise ValueError("Only an MPS can receive deploy_mps tasks.")
            task["node"] = node_name
        elif task_type not in {
            "continue_current", "hold_supply", "stay",
        }:
            raise ValueError(f"Unsupported rollout task: {task_type}")
        task["type"] = task_type
        normalized.append(task)
    return normalized


def _evaluate_rollout(
    source: DSRDashboardBackend,
    scenario: Dict[str, Any],
    horizon_steps: int,
) -> Dict[str, Any]:
    candidate = copy.deepcopy(source)
    unit_name = scenario.get("unit")
    unit_name = str(unit_name) if unit_name is not None else None
    tasks = _normalize_rollout_tasks(
        source,
        unit_name,
        scenario.get("tasks", []),
    )
    switch_commands = {
        str(name): str(state).lower()
        for name, state in dict(scenario.get("switch_commands", {})).items()
    }
    trajectory = []
    task_index = 0
    try:
        for offset in range(horizon_steps):
            if candidate.game_score.ended:
                trajectory.extend(
                    [_rollout_weighted_load(candidate)]
                    * (horizon_steps - len(trajectory))
                )
                break
            commands: Dict[str, Dict[str, Any]] = {}
            for name, unit in candidate.mobile_units.items():
                if name == unit_name:
                    command, task_index = _rollout_selected_command(
                        candidate, name, tasks, task_index
                    )
                elif (
                    unit.state in _ROLLOUT_LOCKED_STATES
                ):
                    command = {"type": "continue"}
                elif unit.unit_type == "MPS" and unit.state == "Supplying":
                    context = candidate.mps_supply_context(name)
                    command = {
                        "type": (
                            "stay"
                            if context.get("grid_connected")
                            else "continue"
                        )
                    }
                else:
                    command = {"type": "stay"}
                commands[name] = command
            switch_operations = list(switch_commands.items()) if offset == 0 \
                else _rollout_remote_recourse(candidate)
            candidate.execute_current_step(
                mobile_commands=commands,
                switch_operations=switch_operations,
                frames_per_step=candidate.physics_frames_per_step,
            )
            trajectory.append(_rollout_weighted_load(candidate))
        return {
            "id": str(scenario.get("id", "")),
            "feasible": True,
            "weighted_energized_trajectory_kw": trajectory,
        }
    except Exception as error:
        return {
            "id": str(scenario.get("id", "")),
            "feasible": False,
            "error": str(error),
            "weighted_energized_trajectory_kw": [],
        }


def _state_payload_for(
    source: DSRDashboardBackend,
    message: str = "",
    include_figure: bool = True,
    include_pixels: bool = True,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "ok": True,
        "message": message,
        "summary": _summary_payload_for(source),
        "game": source.game_payload(),
        "clock": source.clock_payload(),
        "last_event_sequence": source.next_event_sequence - 1,
        "scenario": _scenario_payload_for(source),
        "inspection": _inspection_payload_for(source),
        "tables": _table_payload_for(source),
        "objects": _mobile_command_metadata_for(source),
        "commands": {
            "mobile_units": _mobile_command_metadata_for(source),
            "switches": _switch_command_metadata_for(source),
        },
        "text_interface_url": "/text",
    }
    if include_figure:
        figure = build_grid_figure(source)
        payload["figure"] = json.loads(figure.to_json())
    if include_pixels:
        payload["pixels"] = _pixel_payload_for(source)
    return payload


def _state_payload(message: str = "") -> Dict[str, Any]:
    return _state_payload_for(backend, message)


def _text_state_payload_for(
    source: DSRDashboardBackend,
    message: str = "",
) -> Dict[str, Any]:
    """Return a compact simulator state for text-command clients."""

    inspection = _inspection_payload_for(source)
    patrol_macros = build_patrol_macro_payload(source)
    inspection["patrol_macro_task_count"] = len(patrol_macros["tasks"])
    inspection["patrol_macro_tasks"] = patrol_macros["tasks"]
    inspection["patrol_macro_areas"] = patrol_macros["areas"]

    all_loads = [
        {
            "node": node.name,
            "area": node.area_name,
            "p_kw": node.active_power,
            "q_kvar": node.reactive_power,
            "weight": node.load_weight,
            "energized": node.energized,
        }
        for node in source.nodes.values()
        if node.has_load
    ]
    known_nodes = [
        {
            "name": node.name,
            "area": node.area_name,
            "position": list(node.position),
            "energized": node.energized,
            "has_load": node.has_load,
            "p_kw": node.active_power if node.has_load else 0.0,
            "q_kvar": node.reactive_power if node.has_load else 0.0,
            "weight": node.load_weight if node.has_load else 0.0,
        }
        for node in source.nodes.values()
        if node.knowledge_state(source.event_id) == "known"
    ]
    faults = [
        {
            "name": fault.name,
            "position": list(fault.position),
            "known": fault.is_known(source.event_id),
            "active": fault.active,
            "analyzed": fault.analyzed,
            "analyzed_by": fault.analyzed_by,
            "repair_resource": (
                fault.repair_resource if fault.analyzed else None
            ),
            "remaining_repair_resource": (
                fault.remaining_repair_resource if fault.analyzed else None
            ),
            "areas": list(fault.area_names),
        }
        for fault in source.faults.values()
        if fault.is_known(source.event_id)
    ]
    source_fault = None
    if source.transmission_source is not None:
        source_fault = {
            "name": source.transmission_source.name,
            "node": source.transmission_source.node_name,
            "active": source.transmission_source.fault_active,
            "available": source.transmission_source.available,
            "restoration_time_step": (
                source.transmission_source.restoration_time_step
            ),
            "p_kw": source.transmission_source.active_power,
            "q_kvar": source.transmission_source.reactive_power,
        }

    return {
        "ok": True,
        "message": message,
        "version": "23",
        "summary": _summary_payload_for(source),
        "game": source.game_payload(),
        "clock": source.clock_payload(),
        "last_event_sequence": source.next_event_sequence - 1,
        "inspection": inspection,
        "unknown_regions": source.unknown_region_payload(),
        "source": source_fault,
        "switches": {
            item["name"]: {
                "state": item["state"],
                "node": item["node"],
                "terminal_position": item["terminal_position"],
                "line": item["line"],
                "commandable": not item["disabled"],
                "remote_commandable": item["remote_commandable"],
                "operation_type": item["operation_type"],
                "operation_time_min": item["operation_time_min"],
                "operated_by": item["operated_by"],
                "node_area": source.nodes[item["node"]].area_name,
                "line_area": source.electrical_lines[item["line"]].area_name,
            }
            for item in _switch_command_metadata_for(source)
        },
        "objects": {
            item["name"]: item
            for item in _mobile_command_metadata_for(source)
        },
        "loads": all_loads,
        "known_nodes": known_nodes,
        "faults": faults,
        "power_clusters": source.power_cluster_rows_cache,
        "status_tables": _table_payload_for(source),
        "command_format": COMMAND_SCHEMA,
    }


def _error_payload(
    error: Exception,
    status_code: int = 400,
    *,
    include_current_step: bool = False,
) -> JSONResponse:
    content: Dict[str, Any] = {
        "ok": False,
        "message": str(error),
    }
    if include_current_step:
        content["time_step"] = backend.time_step
        content["game"] = backend.game_payload()
    return JSONResponse(status_code=status_code, content=content)


def _parse_command_payload(
    payload: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, object]], list[list[str]]]:
    mobile_commands = payload.get("mobile_commands", {}) or {}
    switch_command_map = payload.get("switch_commands", {}) or {}

    if not isinstance(mobile_commands, dict):
        raise ValueError("mobile_commands must be a JSON object.")
    if not isinstance(switch_command_map, dict):
        raise ValueError("switch_commands must be a JSON object.")

    switch_operations = []
    for switch_name, target_state in switch_command_map.items():
        normalized_state = str(target_state).strip().lower()
        if normalized_state not in {"opened", "closed"}:
            raise ValueError(
                f"Switch {switch_name} target must be 'opened' or 'closed'."
            )
        switch_operations.append([str(switch_name), normalized_state])
    return mobile_commands, switch_operations


def _command_result_message(result: Dict[str, Any]) -> str:
    command_parts = []
    if result.get("switch_operations"):
        command_parts.append(
            "Switches: "
            + ", ".join(
                f"{name}={state}"
                for name, state in result["switch_operations"]
            )
        )
    if result.get("mobile_commands"):
        command_parts.append(
            "Objects: "
            + "; ".join(
                f"{name}: {command}"
                for name, command in result["mobile_commands"].items()
            )
        )
    return (
        f"Time step {result['time_step']} executed successfully."
        + (" " + " | ".join(command_parts) if command_parts else "")
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    return HTMLResponse(INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/replay/h4", response_class=HTMLResponse)
def h4_replay_index() -> HTMLResponse:
    """Serve the dashboard with the historical H=4 replay preloaded."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | H=4 Replay</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="h4" data-auto-replay="h4-rollout">',
        1,
    )
    return HTMLResponse(html)


@app.get("/replay/open-loop", response_class=HTMLResponse)
def open_loop_replay_index() -> HTMLResponse:
    """Serve the standalone one-shot open-loop controller replay."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | One-Shot Open Loop</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="open-loop" data-auto-replay="one-shot">',
        1,
    )
    return HTMLResponse(html)


@app.get("/replay/greedy", response_class=HTMLResponse)
def greedy_replay_index() -> HTMLResponse:
    """Serve the standalone nearest-first greedy replay."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | Nearest-First Greedy</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="greedy" data-auto-replay="greedy">',
        1,
    )
    return HTMLResponse(html)


@app.get("/replay/greedy-2fault", response_class=HTMLResponse)
def greedy_two_fault_replay_index() -> HTMLResponse:
    """Serve the standalone two-fault nearest-first greedy replay."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | Two-Fault Greedy</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="greedy" data-auto-replay="greedy-2fault">',
        1,
    )
    return HTMLResponse(html)


@app.get("/replay/continuous", response_class=HTMLResponse)
def continuous_replay_index() -> HTMLResponse:
    """Serve the standalone continuous-event one-shot replay."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | Continuous Event Replay</title>",
        1,
    )
    html = html.replace(
        "<title>IEEE 33-Bus DSR Simulator</title>",
        "<title>IEEE 33-Bus DSR Simulator | Known One-Shot</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="continuous" '
        'data-auto-replay="continuous-one-shot">',
        1,
    )
    html = html.replace(
        '<body data-replay-page="simulator" data-scenario="ieee33_v1">',
        '<body data-replay-page="continuous" '
        'data-auto-replay="continuous-one-shot" data-scenario="ieee33_v1">',
        1,
    )
    return HTMLResponse(html)


@app.get("/replay/continuous-rolling", response_class=HTMLResponse)
def continuous_rolling_replay_index() -> HTMLResponse:
    """Serve the standalone continuous event-driven rolling replay."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | Continuous Rolling Replay</title>",
        1,
    )
    html = html.replace(
        "<title>IEEE 33-Bus DSR Simulator</title>",
        "<title>IEEE 33-Bus DSR Simulator | Known Rolling</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="continuous-rolling" '
        'data-auto-replay="continuous-rolling">',
        1,
    )
    html = html.replace(
        '<body data-replay-page="simulator" data-scenario="ieee33_v1">',
        '<body data-replay-page="continuous-rolling" '
        'data-auto-replay="continuous-rolling" data-scenario="ieee33_v1">',
        1,
    )
    return HTMLResponse(html)


@app.get("/replay/continuous-unknown", response_class=HTMLResponse)
def continuous_unknown_replay_index() -> HTMLResponse:
    """Serve continuous rolling patrol with hidden regions."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | Continuous Unknown Patrol</title>",
        1,
    )
    html = html.replace(
        "<title>IEEE 33-Bus DSR Simulator</title>",
        "<title>IEEE 33-Bus DSR Simulator | Unknown Rolling</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="continuous-unknown" '
        'data-auto-replay="continuous-unknown-patrol">',
        1,
    )
    html = html.replace(
        '<body data-replay-page="simulator" data-scenario="ieee33_v1">',
        '<body data-replay-page="continuous-unknown" '
        'data-auto-replay="continuous-unknown-patrol" '
        'data-scenario="ieee33_v1">',
        1,
    )
    return HTMLResponse(html)


def _parent_pf_replay_html(
    *, title: str, page: str, replay_name: str
) -> HTMLResponse:
    """Serve a Parent-Arc + LinDistFlow continuous replay page."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        f"<title>DSR Grid Map v23 | {title}</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        f'<body data-replay-page="{page}" data-auto-replay="{replay_name}">',
        1,
    )
    return HTMLResponse(html)


@app.get("/replay/parent-pf/continuous", response_class=HTMLResponse)
def parent_pf_one_shot_replay_index() -> HTMLResponse:
    return _parent_pf_replay_html(
        title="Parent-Arc + LinDistFlow Known One-Shot",
        page="continuous",
        replay_name="parent-pf-continuous-one-shot",
    )


@app.get("/replay/parent-pf/continuous-rolling", response_class=HTMLResponse)
def parent_pf_known_rolling_replay_index() -> HTMLResponse:
    return _parent_pf_replay_html(
        title="Parent-Arc + LinDistFlow Known Rolling",
        page="continuous-rolling",
        replay_name="parent-pf-continuous-rolling",
    )


@app.get("/replay/parent-pf/continuous-unknown", response_class=HTMLResponse)
def parent_pf_unknown_rolling_replay_index() -> HTMLResponse:
    return _parent_pf_replay_html(
        title="Parent-Arc + LinDistFlow Unknown Rolling",
        page="continuous-unknown",
        replay_name="parent-pf-continuous-unknown",
    )


@app.get("/replay/patrol-coverage", response_class=HTMLResponse)
def patrol_coverage_comparison_index() -> HTMLResponse:
    """Serve the synchronized legacy-versus-coverage patrol comparison."""

    if not PATROL_COVERAGE_COMPARISON_PATH.exists():
        return HTMLResponse(
            "Patrol coverage comparison has not been generated.",
            status_code=500,
        )
    return HTMLResponse(
        PATROL_COVERAGE_COMPARISON_PATH.read_text(encoding="utf-8")
    )


@app.get("/simulator-only", response_class=HTMLResponse)
def simulator_only_index() -> HTMLResponse:
    """Serve the manual DSR interface without controller or replay UI."""

    if not INDEX_PATH.exists():
        return HTMLResponse("index_v23.html is missing.", status_code=500)
    html = INDEX_PATH.read_text(encoding="utf-8")
    html = html.replace(
        "<title>DSR Grid Map v23</title>",
        "<title>DSR Grid Map v23 | Simulator Only</title>",
        1,
    )
    html = html.replace(
        '<body data-replay-page="main">',
        '<body data-replay-page="simulator">',
        1,
    )
    return HTMLResponse(html)


@app.get("/text", response_class=HTMLResponse)
def text_index() -> HTMLResponse:
    if not TEXT_INDEX_PATH.exists():
        return HTMLResponse("text_v23.html is missing.", status_code=500)
    return HTMLResponse(TEXT_INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/docs/{document_name}", response_class=HTMLResponse)
def model_document(document_name: str) -> HTMLResponse:
    path = MODEL_DOCUMENTS.get(str(document_name).strip().lower())
    if path is None:
        return HTMLResponse("Unknown model document.", status_code=404)
    if not path.exists():
        return HTMLResponse("Model document is missing.", status_code=500)
    return HTMLResponse(path.read_text(encoding="utf-8"))


@app.get("/plotly.min.js")
def plotly_script() -> Response:
    return Response(plotly_javascript, media_type="application/javascript")


@app.get("/api/state")
def get_state() -> JSONResponse:
    with backend_lock:
        return JSONResponse(_state_payload(backend.stage))


@app.get("/api/events")
def get_events(after: int = 0) -> JSONResponse:
    """Return committed public events after the requested sequence number."""

    with backend_lock:
        events = [
            event
            for event in backend.event_history
            if int(event["sequence"]) > int(after)
        ]
        return JSONResponse({
            "ok": True,
            "after": int(after),
            "last_sequence": backend.next_event_sequence - 1,
            "events": events,
        })


@app.get("/api/controller/updates")
def get_controller_updates(after: int = 0, limit: int = 1) -> JSONResponse:
    """Return ordered Gurobi/simulator rounds for live following or replay."""

    normalized_limit = max(1, min(int(limit), 10))
    with backend_lock:
        pending = [
            update
            for update in controller_updates
            if int(update["sequence"]) > int(after)
        ][:normalized_limit]
        latest_sequence = controller_next_sequence - 1
        return JSONResponse({
            "ok": True,
            "generation": controller_generation,
            "after": int(after),
            "latest_sequence": latest_sequence,
            "has_more": bool(
                pending and int(pending[-1]["sequence"]) < latest_sequence
            ),
            "updates": pending,
        })


@app.get("/api/controller/replay/results")
def get_controller_replay_results() -> JSONResponse:
    """Return compact summaries for the primary saved replay studies."""

    try:
        names = (
            "greedy-2fault", "greedy", "continuous-unknown-patrol", "continuous-rolling",
            "continuous-one-shot", "one-shot", "h4-rollout", "global",
            "global-no-unknown",
        )
        return JSONResponse({
            "ok": True,
            "results": [
                (
                    _saved_continuous_rolling_summary(
                        name, CONTROLLER_REPLAY_RESULTS[name]
                    )
                    if name in {
                        "continuous-rolling", "continuous-unknown-patrol",
                        "parent-pf-continuous-rolling",
                        "parent-pf-continuous-unknown",
                    }
                    else _saved_continuous_result_summary(
                        name, CONTROLLER_REPLAY_RESULTS[name]
                    )
                    if name in {
                        "continuous-one-shot",
                        "parent-pf-continuous-one-shot",
                    }
                    else _saved_controller_result_summary(
                        name, CONTROLLER_REPLAY_RESULTS[name]
                    )
                )
                for name in names
            ],
        })
    except Exception as error:
        return _error_payload(error)


@app.post("/api/controller/replay/load")
async def load_controller_replay(request: Request) -> JSONResponse:
    """Re-execute a saved controller run and expose graphical replay rounds."""

    try:
        payload = await request.json()
        replay_name = str(payload.get("name", "global")).strip().lower()
        result_dir = CONTROLLER_REPLAY_RESULTS.get(replay_name)
        if result_dir is None:
            raise ValueError(
                f"Unknown saved controller replay: {replay_name}"
            )
        with backend_lock:
            if replay_name in {
                "continuous-rolling", "continuous-unknown-patrol",
                "parent-pf-continuous-rolling",
                "parent-pf-continuous-unknown",
            }:
                response = _load_saved_continuous_rolling_replay(result_dir)
                response["result_summary"] = (
                    _saved_continuous_rolling_summary(
                        replay_name, result_dir
                    )
                )
            elif replay_name in {
                "continuous-one-shot",
                "parent-pf-continuous-one-shot",
            }:
                response = _load_saved_continuous_replay(result_dir)
                response["result_summary"] = (
                    _saved_continuous_result_summary(
                        replay_name, result_dir
                    )
                )
            else:
                response = _load_saved_controller_replay(result_dir)
                response["result_summary"] = (
                    _saved_controller_result_summary(
                        replay_name, result_dir
                    )
                )
            return JSONResponse(response)
    except Exception as error:
        return _error_payload(error, include_current_step=True)


@app.post("/api/game/configure")
async def configure_game(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        with backend_lock:
            backend.configure_max_time_step(payload.get("max_time_step"))
            return JSONResponse(_state_payload("Maximum time step updated."))
    except Exception as error:
        return _error_payload(error)


@app.post("/api/load")
async def load_state(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        mode = str(payload.get("mode", "scenario")).strip().lower()
        with backend_lock:
            _reset_controller_updates()
            if mode == "base":
                backend.load_map(render_output=False)
                if payload.get("max_time_step") is not None:
                    backend.configure_max_time_step(
                        payload.get("max_time_step")
                    )
                message = "Base map loaded. Configure the scenario and start."
            elif mode in {"scenario", "faults", "reset"}:
                if payload.get("max_time_step") is not None:
                    backend.configure_max_time_step(
                        payload.get("max_time_step")
                    )
                backend.load_faults_and_unknown(
                    render_output=False,
                    unknown_regions=payload.get("unknown_regions"),
                    fault_scenario=payload.get("fault_scenario"),
                    unit_initial_positions=payload.get(
                        "unit_initial_positions"
                    ),
                )
                message = (
                    "Fault scenario loaded. P0 has been added to the score."
                )
            else:
                raise ValueError(f"Unsupported load mode: {mode}")
            return JSONResponse(_state_payload(message))
    except Exception as error:
        return _error_payload(error)


@app.post("/api/validate-switches")
async def validate_switch_plan(request: Request) -> JSONResponse:
    """Validate provisional Switch actions without committing them."""

    try:
        payload = await request.json()
        switch_command_map = payload.get("switch_commands", {}) or {}
        switch_operations = [
            [switch_name, target_state]
            for switch_name, target_state in switch_command_map.items()
            if target_state in {"opened", "closed"}
        ]

        with backend_lock:
            candidate, normalized = backend.validate_switch_plan(
                switch_operations
            )
            mps_contexts = {
                name: candidate.mps_supply_context(name)
                for name in candidate.mps_units
            }
            return JSONResponse({
                "ok": True,
                "message": (
                    "Switch plan validated. MPS load requirements now use "
                    "the provisional topology."
                ),
                "switch_operations": [list(item) for item in normalized],
                "mps_contexts": mps_contexts,
                "grid_available": bool(
                    candidate.transmission_source is not None
                    and candidate.transmission_source.available
                ),
                "candidate_switches": _switch_command_metadata_for(candidate),
                "predicted_summary": _summary_payload_for(candidate),
                "predicted_restored_active_power_kw": (
                    candidate.restored_active_power()
                ),
                "predicted_weighted_energized_kw": sum(
                    node.active_power * node.load_weight
                    for node in candidate.nodes.values()
                    if node.has_load and node.energized
                ),
            })
    except Exception as error:
        return _error_payload(error)


@app.post("/api/planning/rollouts")
async def evaluate_planning_rollouts(request: Request) -> JSONResponse:
    """Evaluate public, non-committing look-ahead policies on isolated copies."""

    try:
        payload = await request.json()
        horizon_steps = max(1, min(int(payload.get("horizon_steps", 4)), 5))
        scenarios = list(payload.get("scenarios", []))
        if len(scenarios) > 160:
            raise ValueError("At most 160 rollout scenarios are allowed.")
        with backend_lock:
            results = [
                _evaluate_rollout(backend, scenario, horizon_steps)
                for scenario in scenarios
            ]
            return JSONResponse({
                "ok": True,
                "horizon_steps": horizon_steps,
                "minutes_per_step": backend.minutes_per_step,
                "current_weighted_energized_kw": _rollout_weighted_load(
                    backend
                ),
                "results": results,
            })
    except Exception as error:
        return _error_payload(error)


@app.get("/api/planning/patrol-macros")
def get_patrol_macro_tasks() -> JSONResponse:
    """Return finite public patrol tasks for the current Pixel knowledge."""

    with backend_lock:
        return JSONResponse({
            "ok": True,
            **build_patrol_macro_payload(backend),
        })


@app.post("/api/planning/travel-times")
async def get_planning_travel_times(request: Request) -> JSONResponse:
    """Return exact public road travel times between planning locations."""

    try:
        payload = await request.json()
        raw_locations = dict(payload.get("locations", {}))
        if not raw_locations or len(raw_locations) > 80:
            raise ValueError("locations must contain between 1 and 80 entries.")
        with backend_lock:
            locations = {
                str(name): tuple(int(value) for value in position)
                for name, position in raw_locations.items()
            }
            for name, position in locations.items():
                if len(position) != 2 or not backend.graph.contains(position):
                    raise ValueError(f"Invalid planning location {name}: {position}")
            requested_units = payload.get("units") or [
                name for name, unit in backend.mobile_units.items()
                if unit.unit_type in {"RC", "Scott", "MPS"}
            ]
            matrices = planning_travel_times_for(
                backend, locations, [str(name) for name in requested_units]
            )
            return JSONResponse({
                "ok": True,
                "clock": backend.clock_payload(),
                "locations": {
                    name: list(position) for name, position in locations.items()
                },
                "travel_time_min": matrices,
            })
    except Exception as error:
        return _error_payload(error)


@app.post("/api/planning/mps-contexts")
async def get_planning_mps_contexts(request: Request) -> JSONResponse:
    """Evaluate public MPS connection candidates without changing live state."""

    try:
        payload = await request.json()
        requests = list(payload.get("candidates", []))
        if not requests or len(requests) > 40:
            raise ValueError("candidates must contain between 1 and 40 entries.")
        with backend_lock:
            switch_commands = dict(payload.get("switch_commands", {}))
            provisional, _ = backend.validate_switch_plan(
                list(switch_commands.items())
            )
            results = []
            for item in requests:
                unit_name = str(item.get("unit", ""))
                node_name = str(item.get("node", ""))
                if unit_name not in provisional.mps_units:
                    raise KeyError(f"Unknown MPS: {unit_name}")
                if node_name not in provisional.nodes:
                    raise KeyError(f"Unknown Node: {node_name}")
                if provisional.nodes[node_name].knowledge_state(
                    provisional.event_id
                ) != "known":
                    raise ValueError(
                        f"MPS planning candidate {node_name} is not public knowledge."
                    )

                candidate = copy.deepcopy(provisional)
                mps = candidate.mps_units[unit_name]
                mps.disconnect()
                mps.position = candidate.nodes[node_name].position
                context = candidate.mps_supply_context(unit_name)
                load_by_area: Dict[str, float] = {}
                for connected_node_name in context.get("nodes", []):
                    node = candidate.nodes[connected_node_name]
                    if not node.has_load or node.energized:
                        continue
                    area_name = node.area_name or "UNASSIGNED"
                    load_by_area[area_name] = (
                        load_by_area.get(area_name, 0.0)
                        + node.active_power * node.load_weight
                    )
                capacity_feasible = bool(
                    float(context.get("load_p", 0.0)) <= mps.p_limit + 1e-9
                    and (
                        float(context.get("load_p", 0.0)) ** 2
                        + float(context.get("load_q", 0.0)) ** 2
                    ) ** 0.5 <= mps.s_limit + 1e-9
                )
                results.append({
                    "unit": unit_name,
                    "node": node_name,
                    "position": list(candidate.nodes[node_name].position),
                    "available": bool(context.get("available")),
                    "capacity_feasible": capacity_feasible,
                    "grid_connected": bool(context.get("grid_connected")),
                    "load_p": float(context.get("load_p", 0.0)),
                    "load_q": float(context.get("load_q", 0.0)),
                    "weighted_unenergized_load_by_area": load_by_area,
                    "weighted_unenergized_load": sum(load_by_area.values()),
                })
            return JSONResponse({"ok": True, "results": results})
    except Exception as error:
        return _error_payload(error)


@app.post("/api/execute")
async def execute_current_step(request: Request) -> JSONResponse:
    try:
        payload = await request.json()
        mobile_commands, switch_operations = _parse_command_payload(payload)

        with backend_lock:
            result = backend.execute_current_step(
                mobile_commands=mobile_commands,
                switch_operations=switch_operations,
                frames_per_step=backend.physics_frames_per_step,
                collect_frame_snapshots=True,
            )
            frame_snapshots = result.pop("frame_snapshots")
            animation_frames = [
                {
                    "frame": index,
                    "figure": json.loads(
                        build_grid_figure(snapshot).to_json()
                    ),
                }
                for index, snapshot in enumerate(frame_snapshots, start=1)
            ]
            state = _state_payload(_command_result_message(result))
            state["animation_frames"] = animation_frames
            state["events"] = result.get("events", [])
            state["execution"] = {
                "actual_repair_resource": result.get(
                    "actual_repair_resource", {}
                ),
                "switch_operations": result.get("switch_operations", []),
                "mobile_commands": result.get("mobile_commands", {}),
            }
            return JSONResponse(state)
    except Exception as error:
        return _error_payload(error, include_current_step=True)


@app.post("/api/controller/execute")
async def execute_controller_step(request: Request) -> JSONResponse:
    """Execute and retain one externally optimized round for the web UI."""

    try:
        payload = await request.json()
        mobile_commands, switch_operations = _parse_command_payload(payload)
        controller = payload.get("controller", {}) or {}
        if not isinstance(controller, dict):
            raise ValueError("controller metadata must be a JSON object.")

        with backend_lock:
            result = backend.execute_current_step(
                mobile_commands=mobile_commands,
                switch_operations=switch_operations,
                frames_per_step=backend.physics_frames_per_step,
                collect_frame_snapshots=True,
            )
            frame_snapshots = result.pop("frame_snapshots")
            animation_frames = [
                {
                    "frame": index,
                    "figure": json.loads(
                        build_grid_figure(snapshot).to_json()
                    ),
                }
                for index, snapshot in enumerate(frame_snapshots, start=1)
            ]
            message = _command_result_message(result)
            graphical_state = _state_payload(message)
            update = _store_controller_update({
                "round": int(controller.get("round", backend.time_step)),
                "observation": controller.get("observation", {}),
                "plan": controller.get("plan", {}),
                "validation": str(
                    controller.get("validation", "validated")
                ),
                "command": {
                    "mobile_commands": mobile_commands,
                    "switch_commands": {
                        name: state for name, state in switch_operations
                    },
                },
                "animation_frames": animation_frames,
                "events": result.get("events", []),
                "state": graphical_state,
            })
            return JSONResponse({
                "ok": True,
                "message": message,
                "controller_sequence": update["sequence"],
                "controller_generation": update["generation"],
                "state": _text_state_payload_for(backend, backend.stage),
                "events": result.get("events", []),
                "execution": {
                    "actual_repair_resource": result.get(
                        "actual_repair_resource", {}
                    ),
                    "switch_operations": result.get(
                        "switch_operations", []
                    ),
                    "mobile_commands": result.get("mobile_commands", {}),
                },
            })
    except Exception as error:
        return _error_payload(error, include_current_step=True)


@app.get("/api/download/state.json")
def download_current_state() -> Response:
    """Download the current environment output as formatted JSON."""

    with backend_lock:
        text = json.dumps(
            _text_state_payload_for(backend, backend.stage),
            ensure_ascii=False,
            indent=2,
        )
        filename = f"dsr_state_T{backend.time_step:03d}.json"
        return Response(
            text,
            media_type="application/json; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            },
        )


@app.get("/api/text/state")
def get_text_state() -> JSONResponse:
    with backend_lock:
        return JSONResponse(_text_state_payload_for(backend, backend.stage))


@app.get("/api/text/state.txt")
def get_text_state_plain() -> Response:
    with backend_lock:
        text = json.dumps(
            _text_state_payload_for(backend, backend.stage),
            ensure_ascii=False,
            indent=2,
        )
        return Response(text, media_type="text/plain; charset=utf-8")


@app.post("/api/text/validate")
async def validate_text_command(request: Request) -> JSONResponse:
    """Validate a complete JSON command batch without changing live state."""

    try:
        payload = await request.json()
        mobile_commands, switch_operations = _parse_command_payload(payload)
        with backend_lock:
            candidate = backend.validate_command_batch(
                mobile_commands=mobile_commands,
                switch_operations=switch_operations,
            )
            return JSONResponse({
                "ok": True,
                "message": "Command batch is executable.",
                "live_time_step": backend.time_step,
                "predicted_state": _text_state_payload_for(
                    candidate,
                    "Predicted state after one time step.",
                ),
                "predicted_events": list(candidate.last_step_events),
            })
    except Exception as error:
        return _error_payload(error, include_current_step=True)


@app.post("/api/text/execute")
async def execute_text_command(request: Request) -> JSONResponse:
    """Execute a complete JSON command batch atomically."""

    try:
        payload = await request.json()
        mobile_commands, switch_operations = _parse_command_payload(payload)
        with backend_lock:
            result = backend.execute_current_step(
                mobile_commands=mobile_commands,
                switch_operations=switch_operations,
                frames_per_step=backend.physics_frames_per_step,
                collect_frame_snapshots=False,
            )
            result.pop("frame_snapshots", None)
            return JSONResponse({
                "ok": True,
                "message": _command_result_message(result),
                "state": _text_state_payload_for(backend, backend.stage),
                "events": result.get("events", []),
                "execution": {
                    "actual_repair_resource": result.get(
                        "actual_repair_resource", {}
                    ),
                    "switch_operations": result.get(
                        "switch_operations", []
                    ),
                    "mobile_commands": result.get("mobile_commands", {}),
                },
            })
    except Exception as error:
        return _error_payload(error, include_current_step=True)


@app.post("/api/shutdown")
def shutdown() -> JSONResponse:
    global server_instance
    if server_instance is not None:
        server_instance.should_exit = True
    return JSONResponse({"ok": True, "message": "Server shutdown requested."})


def run_server(host: str, port: int, open_browser: bool) -> None:
    global server_instance
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server_instance = uvicorn.Server(config)
    if open_browser:
        threading.Timer(
            1.0,
            lambda: webbrowser.open(f"http://{host}:{port}"),
        ).start()
    server_instance.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DSR Grid Map v23 locally.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8050, type=int)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the browser automatically.",
    )
    arguments = parser.parse_args()
    run_server(
        host=arguments.host,
        port=arguments.port,
        open_browser=not arguments.no_browser,
    )


if __name__ == "__main__":
    main()
