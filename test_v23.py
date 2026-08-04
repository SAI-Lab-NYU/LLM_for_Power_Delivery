from __future__ import annotations

import asyncio
import copy
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mpl_"))
PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT))

from simulator_core_v23 import DSRDashboardBackend, GameScore


def commands(backend, **overrides):
    result = {name: {"type": "continue"} for name in backend.mobile_units}
    result.update(overrides)
    return result


def clear_all_faults(backend):
    for fault in backend.faults.values():
        fault.active = False
        fault.remaining_repair_resource = 0
        fault.pixel.remove_fault(fault.name)
    for pixel in backend.graph.pixels.values():
        pixel.mark_inspected(backend.event_id)
    for switch in backend.switches.values():
        switch.update_knowledge(backend.nodes, backend.event_id)
    for line in backend.electrical_lines.values():
        line.update_state()


# Weighted GameScore behavior and per-load continuity.
score = GameScore(5)
score.start(100, ["LoadA"])
assert score.score == 100
score.record(1, 150, ["LoadA", "LoadB"])
assert score.score == 250
# The weighted metric may decrease, for example after energizing a negative-P load.
# This does not lock the score while every previously restored load remains on.
score.record(2, 140, ["LoadA", "LoadB", "NegativeLoad"])
assert score.score == 390 and not score.locked
# Losing any previously energized load resets and permanently locks the score.
score.record(3, 999, ["LoadB", "NegativeLoad"])
assert score.score == 0 and score.locked and score.violation_step == 3
assert score.violation_loads == ["LoadA"]
score.record(4, 1200, ["LoadA", "LoadB", "NegativeLoad"])
assert score.score == 0

# Multiple unknown rectangles and normalization.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(4)
backend.load_faults_and_unknown(
    render_output=False,
    unknown_regions=[
        {"name": "A", "lower_left": [-13, -17], "upper_right": [3, 3]},
        {"name": "B", "lower_left": [8, 8], "upper_right": [-2, -4]},
        {"name": "C", "lower_left": [10, -30], "upper_right": [30, -10]},
    ],
)
regions = backend.unknown_region_payload()
assert len(regions) == 3
assert regions[1]["lower_left"] == [-2, -4]
assert regions[1]["upper_right"] == [8, 8]
assert not backend.graph.get_pixel((0, 0)).is_inspected(backend.event_id)
assert not backend.graph.get_pixel((15, -18)).is_inspected(backend.event_id)
assert backend.game_score.started and backend.game_score.power_history == [0.0]
assert backend.game_score.energized_load_history == [[]]

# Normal scoring and monotonicity violation.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(5)
backend.load_faults_and_unknown(render_output=False)
result = backend.execute_current_step(
    mobile_commands=commands(backend, MPS1={"type": "supply", "p": 800, "q": 440}),
    switch_operations=[],
    frames_per_step=1,
)
assert backend.time_step == 1
assert backend.restored_active_power() == 800
assert backend.weighted_restored_load() == 2990
assert backend.game_score.score == 2990
assert backend.game_score.energized_load_history[-1] == ["N2", "N5", "N6", "N7"]
backend.execute_current_step(
    mobile_commands=commands(backend, MPS1={"type": "stay"}),
    switch_operations=[],
    frames_per_step=1,
)
assert backend.time_step == 2
assert backend.restored_active_power() == 0
assert backend.game_score.score == 0 and backend.game_score.locked
assert backend.game_score.violation_loads == ["N2", "N5", "N6", "N7"]
backend.execute_current_step(
    mobile_commands=commands(backend, MPS1={"type": "supply", "p": 800, "q": 440}),
    switch_operations=[],
    frames_per_step=1,
)
assert backend.game_score.score == 0

# Maximum-step ending and command lock.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(2)
backend.load_faults_and_unknown(render_output=False)
for _ in range(2):
    backend.execute_current_step(
        mobile_commands=commands(backend),
        switch_operations=[],
        frames_per_step=1,
    )
assert backend.time_step == 2 and backend.game_score.ended
try:
    backend.execute_current_step(commands(backend), [], 1)
except RuntimeError as error:
    assert "ended" in str(error).lower()
else:
    raise AssertionError("Ended game accepted a command")

# Early completion and projected score.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(5)
backend.load_faults_and_unknown(render_output=False, unknown_regions=[])
clear_all_faults(backend)
backend.transmission_source.restoration_time_step = 1
backend.transmission_source.update_availability(backend.time_step)
backend._refresh_derived_states()
total_p = backend.total_active_load()
total_weighted = backend.total_weighted_load()
assert backend.game_score.score == 0
backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[["S1", "closed"], ["S3", "closed"], ["S4", "closed"], ["S5", "closed"]],
    frames_per_step=1,
)
assert backend.time_step == 1
assert backend.all_faults_cleared()
assert backend.all_loads_restored()
assert backend.game_score.ended
assert backend.game_score.projected_steps == 4
assert abs(backend.game_score.score - total_weighted * 5) < 1e-9
assert backend.game_payload()["total_active_load"] == total_p
assert backend.game_payload()["total_weighted_load"] == total_weighted

# Signed loads and grid absorption.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.load_map(render_output=False)
for node in backend.nodes.values():
    if node.has_load:
        node.load = (-abs(node.active_power), node.reactive_power, node.load_weight)
backend._refresh_derived_states()
assert backend.transmission_source.active_power == -2720

# Text validation uses a copy and rejected execution preserves time.
import simulator_web_v23 as web

route_paths = {route.path for route in web.app.routes}
for required in {
    "/text", "/api/text/state", "/api/text/state.txt",
    "/api/text/validate", "/api/text/execute",
}:
    assert required in route_paths

async def api_tests():
    # Directly call async route functions with a tiny request stub to avoid
    # dependency-version coupling in TestClient.
    class RequestStub:
        def __init__(self, payload):
            self.payload = payload
        async def json(self):
            return copy.deepcopy(self.payload)

    await web.load_state(RequestStub({"mode": "base", "max_time_step": 3}))
    response = await web.load_state(RequestStub({
        "mode": "scenario",
        "max_time_step": 3,
        "unknown_regions": [{
            "name": "Only",
            "lower_left": [-13, -17],
            "upper_right": [3, 3],
        }],
    }))
    assert response.status_code == 200
    assert web.backend.time_step == 0

    valid = {
        "switch_commands": {},
        "mobile_commands": {"MPS1": {"type": "supply", "p": 800, "q": 440}},
    }
    response = await web.validate_text_command(RequestStub(valid))
    body = json.loads(response.body)
    assert response.status_code == 200 and body["ok"]
    assert body["live_time_step"] == 0 and web.backend.time_step == 0
    assert body["predicted_state"]["game"]["time_step"] == 1
    assert body["predicted_state"]["game"]["weighted_restored_load"] == 2990
    assert body["predicted_state"]["game"]["score"] == 2990

    response = await web.execute_text_command(RequestStub(valid))
    body = json.loads(response.body)
    assert response.status_code == 200 and body["ok"]
    assert web.backend.time_step == 1

    invalid = {
        "switch_commands": {},
        "mobile_commands": {"MPS1": {"type": "supply", "p": 9999, "q": 0}},
    }
    response = await web.execute_text_command(RequestStub(invalid))
    body = json.loads(response.body)
    assert response.status_code == 400 and not body["ok"]
    assert body["time_step"] == 1 and web.backend.time_step == 1

asyncio.run(api_tests())

print("All V23-HF4 backend and text-interface tests passed.")

# Fault-analysis lifecycle: discovery pauses movement, locks one complete step,
# hides repair resources, and then resumes the paused Move command.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(20)
backend.load_faults_and_unknown(render_output=False)
backend.execute_current_step(
    mobile_commands=commands(
        backend,
        PAC1={"type": "move", "waypoints": [[-4, 0]]},
    ),
    switch_operations=[],
    frames_per_step=4,
)
backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[],
    frames_per_step=4,
)
pac = backend.pacs["PAC1"]
fault_b = backend.faults["B"]
assert fault_b.is_known(backend.event_id)
assert not fault_b.analyzed
assert pac.state == "Analyzing" and pac.analysis_locked
locked_position = pac.position
locked_path = list(pac.planned_path)
row_b = next(row for row in backend.fault_rows() if row[0] == "B")
assert row_b[4] == "Hidden" and row_b[5] == "Hidden"

# A locked object cannot accept a replacement command, and rejection is atomic.
locked_time = backend.time_step
try:
    backend.execute_current_step(
        mobile_commands=commands(
            backend,
            PAC1={"type": "move", "waypoints": [[0, 0]]},
        ),
        switch_operations=[],
        frames_per_step=1,
    )
except ValueError as error:
    assert "locked" in str(error).lower()
else:
    raise AssertionError("Analysis-locked PAC accepted a Move command")
assert backend.time_step == locked_time
assert backend.pacs["PAC1"].position == locked_position
assert backend.pacs["PAC1"].planned_path == locked_path

# Continue represents the mandatory automatic analysis step. No movement occurs.
result = backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[],
    frames_per_step=4,
)
pac = backend.pacs["PAC1"]
assert pac.position == locked_position
fault_b = backend.faults["B"]
assert fault_b.analyzed
assert not pac.analysis_locked
assert pac.state == "Moving"
assert pac.planned_path == locked_path
assert "Automatic fault analysis" in result["mobile_commands"]["PAC1"]
row_b = next(row for row in backend.fault_rows() if row[0] == "B")
assert row_b[4] == str(fault_b.repair_resource)
assert row_b[5] == str(fault_b.remaining_repair_resource)

# A following Continue resumes the interrupted route.
backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[],
    frames_per_step=4,
)
assert backend.pacs["PAC1"].position != locked_position

# A known but unanalyzed fault triggers analysis when RC/PAC arrives exactly.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(10)
backend.load_faults_and_unknown(render_output=False)
fault_a = backend.faults["A"]
crew = backend.repair_crews["RC1"]
crew.position = fault_a.position
try:
    backend.execute_current_step(
        mobile_commands=commands(
            backend,
            RC1={"type": "repair", "fault": "A", "amount": 1},
        ),
        switch_operations=[],
        frames_per_step=1,
    )
except ValueError as error:
    assert "locked" in str(error).lower()
else:
    raise AssertionError("RC repaired a known but unanalyzed fault")
assert not fault_a.analyzed
result = backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[],
    frames_per_step=1,
)
fault_a = backend.faults["A"]
assert fault_a.analyzed
assert backend.maximum_repair_amount("RC1", "A") > 0
assert "Automatic fault analysis" in result["mobile_commands"]["RC1"]

# MPS inherits the analysis interface but has it disabled: discovery does not
# stop movement and does not reveal repair-resource requirements.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(20)
backend.load_faults_and_unknown(render_output=False)
for step_index in range(5):
    mps_command = (
        {"type": "move", "waypoints": [[-4, -4]]}
        if step_index == 0
        else {"type": "continue"}
    )
    backend.execute_current_step(
        mobile_commands=commands(backend, MPS1=mps_command),
        switch_operations=[],
        frames_per_step=4,
    )
mps = backend.mps_units["MPS1"]
fault_b = backend.faults["B"]
assert not mps.fault_analysis_enabled
assert not mps.analysis_locked
assert mps.state == "Moving"
assert fault_b.is_known(backend.event_id) and not fault_b.analyzed

# Graphical download support and text output contain the complete status state.
route_paths = {route.path for route in web.app.routes}
assert "/api/download/state.json" in route_paths
response = web.download_current_state()
assert response.status_code == 200
assert "attachment" in response.headers["content-disposition"].lower()
downloaded_state = json.loads(response.body)
assert "game" in downloaded_state
assert "weighted_restored_load" in downloaded_state["game"]
assert "weighted_load_history" in downloaded_state["game"]
assert "energized_load_history" in downloaded_state["game"]
assert "status_tables" in downloaded_state
assert "faults" in downloaded_state["status_tables"]

html = (PROJECT / "index_v23.html").read_text(encoding="utf-8")
assert 'id="download-state"' in html
assert 'id="download-commands"' in html
assert "Download Pending Commands JSON" in html
assert "unit.analysis_locked" in html

print("All V23 fault-analysis and download tests passed.")

# Regression: presentation and text interfaces must not reveal fault-resource
# requirements before RC/PAC analysis, and completely unknown faults must not
# appear in environment-output payloads.
visibility_backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
visibility_backend.configure_max_time_step(10)
visibility_backend.load_faults_and_unknown(render_output=False)
fault_a = visibility_backend.faults["A"]
fault_b = visibility_backend.faults["B"]
assert fault_a.is_known(visibility_backend.event_id) and not fault_a.analyzed
assert not fault_b.is_known(visibility_backend.event_id) and not fault_b.analyzed

visibility_text_state = web._text_state_payload_for(visibility_backend)
assert [fault["name"] for fault in visibility_text_state["faults"]] == ["A"]
text_fault_a = visibility_text_state["faults"][0]
assert text_fault_a["repair_resource"] is None
assert text_fault_a["remaining_repair_resource"] is None
assert all(
    row[0] != "B"
    for row in visibility_text_state["status_tables"]["faults"]["rows"]
)
assert all(
    not (row[0] == "Fault" and row[1] == "B")
    for row in visibility_text_state["status_tables"]["elements"]["rows"]
)

visibility_graph_state = web._state_payload_for(
    visibility_backend,
    include_figure=True,
    include_pixels=True,
)
assert visibility_graph_state["pixels"]["-4,-4"]["faults"] == []
fault_a_traces = [
    trace
    for trace in visibility_graph_state["figure"]["data"]
    if trace.get("name") == "Fault A"
]
assert len(fault_a_traces) == 1
fault_a_hover = fault_a_traces[0]["hovertemplate"]
assert "Required repair resource: Hidden" in fault_a_hover
assert "Remaining repair resource: Hidden" in fault_a_hover
assert "Required repair resource: 10" not in fault_a_hover
assert "Remaining repair resource: 10" not in fault_a_hover

# Once analysis is complete, the same graphical trace may reveal the values.
fault_a.analyzed = True
fault_a.analyzed_by = "RC1"
analyzed_figure = web.build_grid_figure(visibility_backend)
analyzed_trace = next(
    trace for trace in analyzed_figure.data if trace.name == "Fault A"
)
assert "Required repair resource: 10" in analyzed_trace.hovertemplate
assert "Remaining repair resource: 10" in analyzed_trace.hovertemplate

async def information_hiding_api_tests():
    class RequestStub:
        def __init__(self, payload):
            self.payload = payload

        async def json(self):
            return copy.deepcopy(self.payload)

    response = await web.load_state(RequestStub({
        "mode": "reset",
    }))
    assert response.status_code == 200

    state_response = web.get_text_state()
    state_body = json.loads(state_response.body)
    assert [fault["name"] for fault in state_body["faults"]] == ["A"]
    assert state_body["faults"][0]["repair_resource"] is None
    assert state_body["faults"][0]["remaining_repair_resource"] is None

    plain_response = web.get_text_state_plain()
    plain_body = json.loads(plain_response.body)
    assert [fault["name"] for fault in plain_body["faults"]] == ["A"]
    assert plain_body["faults"][0]["repair_resource"] is None

    download_response = web.download_current_state()
    download_body = json.loads(download_response.body)
    assert [fault["name"] for fault in download_body["faults"]] == ["A"]
    assert download_body["faults"][0]["repair_resource"] is None

    command = {
        "switch_commands": {},
        "mobile_commands": {
            name: {"type": "continue"}
            for name in web.backend.mobile_units
        },
    }
    predicted_response = await web.validate_text_command(RequestStub(command))
    predicted_body = json.loads(predicted_response.body)
    assert predicted_response.status_code == 200
    predicted_faults = predicted_body["predicted_state"]["faults"]
    assert [fault["name"] for fault in predicted_faults] == ["A"]
    assert predicted_faults[0]["repair_resource"] is None
    assert predicted_faults[0]["remaining_repair_resource"] is None

asyncio.run(information_hiding_api_tests())

print("All V23 fault-information-hiding regression tests passed.")


# V23-HF4 absolute-power weighted score regression checks.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(5)
backend.load_faults_and_unknown(render_output=False)
payload = backend.game_payload()
assert payload["score_basis"] == "sum_over_time(sum_over_energized_loads(abs(P)*W))"
assert payload["weighted_restored_load"] == 0
assert payload["energized_loads"] == []
assert payload["violation_loads"] == []

# Direct score regression: a lower weighted metric alone is legal.
score = GameScore(4)
score.start(500, ["L1"])
score.record(1, 400, ["L1", "L2"])
assert score.score == 900 and not score.locked
score.record(2, 700, ["L2"])
assert score.score == 0 and score.locked
assert score.violation_loads == ["L1"]

print("All V23-HF4 absolute-power weighted scoring regression tests passed.")


# V23-HF4 negative-active-power scoring regression.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.load_map(render_output=False)
for node in backend.nodes.values():
    node.energized = False
negative_load = backend.nodes["N12"]
negative_load.load = (-175.0, negative_load.reactive_power, negative_load.load_weight)
negative_load.energized = True
assert backend.restored_active_power() == -175.0
assert backend.weighted_restored_load() == 875.0
expected_total = sum(
    abs(node.active_power) * node.load_weight
    for node in backend.nodes.values()
    if node.has_load
)
assert backend.total_weighted_load() == expected_total

backend.game_score.start(
    restored_active_power=backend.restored_active_power(),
    energized_loads=backend.energized_load_names(),
    weighted_restored_load=backend.weighted_restored_load(),
)
assert backend.game_score.score == 875.0
assert backend.game_score.current_power == -175.0
assert backend.game_score.current_weighted_load == 875.0
summary = web._summary_payload_for(backend)
assert summary["weighted_energized_load"] == "875"

print("All V23-HF4 negative-active-power scoring tests passed.")
