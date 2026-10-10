from __future__ import annotations

import argparse
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

from simulator_core_v23 import DSRDashboardBackend
from simulator_plotly_v23 import build_grid_figure


APP_DIR = Path(__file__).resolve().parent
INDEX_PATH = APP_DIR / "index_v23.html"
TEXT_INDEX_PATH = APP_DIR / "text_v23.html"

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

app = FastAPI(title="DSR Grid Map v23")


TABLE_DEFINITIONS = {
    "sources": {
        "title": "Power Sources",
        "headers": [
            "Source", "Node", "Status", "Restore Step", "P/kW",
            "Q/kVar", "Isolated Switches",
        ],
    },
    "mps": {
        "title": "MPS States",
        "headers": [
            "MPS", "State", "Position", "Energy/kW.step", "P Limit",
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
            "Node-side Area", "Line-side Area",
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
        "Scout": {"type": "stay"},
        "MPS1": {"type": "supply", "p": 800, "q": 440},
    },
}


def _summary_payload_for(source: DSRDashboardBackend) -> Dict[str, Any]:
    known_faults = sum(
        fault.active and fault.is_known(source.event_id)
        for fault in source.faults.values()
    )
    if (
        source.transmission_source is not None
        and source.transmission_source.fault_active
    ):
        known_faults += 1

    energized_active_power = source.restored_active_power()
    weighted_energized_load = sum(
        node.active_power * node.load_weight
        for node in source.nodes.values()
        if node.has_load and node.energized
    )

    return {
        "stage": source.stage,
        "time_step": source.time_step,
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
            "repair_target": unit.repair_fault_name,
            "fault_analysis_enabled": unit.fault_analysis_enabled,
            "analysis_locked": unit.analysis_locked,
            "analysis_faults": list(unit.analysis_fault_names),
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
    return [
        {
            "name": switch.name,
            "state": switch.state,
            "node": switch.node_name,
            "line": switch.line_name,
            "disabled": switch.state == "unknown",
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
            "faults": sorted(pixel.active_fault_ids),
        }
        for (x, y), pixel in source.graph.pixels.items()
    }


def _scenario_payload_for(source: DSRDashboardBackend) -> Dict[str, Any]:
    return {
        "unknown_regions": source.unknown_region_payload(),
        "can_edit": not source.game_score.started,
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
        "scenario": _scenario_payload_for(source),
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

    all_loads = [
        {
            "node": node.name,
            "p_kw": node.active_power,
            "q_kvar": node.reactive_power,
            "weight": node.load_weight,
            "energized": node.energized,
        }
        for node in source.nodes.values()
        if node.has_load
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
        "unknown_regions": source.unknown_region_payload(),
        "source": source_fault,
        "switches": {
            item["name"]: {
                "state": item["state"],
                "node": item["node"],
                "line": item["line"],
                "commandable": not item["disabled"],
            }
            for item in _switch_command_metadata_for(source)
        },
        "objects": {
            item["name"]: item
            for item in _mobile_command_metadata_for(source)
        },
        "loads": all_loads,
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


@app.get("/text", response_class=HTMLResponse)
def text_index() -> HTMLResponse:
    if not TEXT_INDEX_PATH.exists():
        return HTMLResponse("text_v23.html is missing.", status_code=500)
    return HTMLResponse(TEXT_INDEX_PATH.read_text(encoding="utf-8"))


@app.get("/plotly.min.js")
def plotly_script() -> Response:
    return Response(plotly_javascript, media_type="application/javascript")


@app.get("/api/state")
def get_state() -> JSONResponse:
    with backend_lock:
        return JSONResponse(_state_payload(backend.stage))


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
            if mode == "base":
                backend.load_map(render_output=False)
                if payload.get("max_time_step") is not None:
                    backend.configure_max_time_step(
                        payload.get("max_time_step")
                    )
                message = "Base map loaded. Configure N, then load the file-defined scenario."
            elif mode in {"scenario", "faults", "reset"}:
                # A scenario load is a complete restart. Preserve the current N
                # unless the request supplies a replacement value, then rebuild
                # every runtime object before loading faults and unknown Pixels.
                requested_max_time_step = payload.get(
                    "max_time_step",
                    backend.game_score.max_time_step,
                )
                backend.load_map(render_output=False)
                backend.configure_max_time_step(requested_max_time_step)
                backend.load_faults_and_unknown(
                    render_output=False,
                    unknown_regions=payload.get("unknown_regions"),
                )
                message = (
                    "Fault scenario reset to time step 0. "
                    "P0 has been added to the score."
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
            })
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
                frames_per_step=4,
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
            return JSONResponse(state)
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
                frames_per_step=1,
                collect_frame_snapshots=False,
            )
            result.pop("frame_snapshots", None)
            return JSONResponse({
                "ok": True,
                "message": _command_result_message(result),
                "state": _text_state_payload_for(backend, backend.stage),
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
