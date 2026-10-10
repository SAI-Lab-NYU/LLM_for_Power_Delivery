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


# GameScore unit behavior.
score = GameScore(5)
score.start(100)
assert score.score == 100
score.record(1, 150)
assert score.score == 250
score.record(2, 140)
assert score.score == 0 and score.locked and score.violation_step == 2
score.record(3, 999)
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
assert backend.game_score.score == 800
backend.execute_current_step(
    mobile_commands=commands(backend, MPS1={"type": "stay"}),
    switch_operations=[],
    frames_per_step=1,
)
assert backend.time_step == 2
assert backend.restored_active_power() == 0
assert backend.game_score.score == 0 and backend.game_score.locked
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
assert abs(backend.game_score.score - total_p * 5) < 1e-9

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

print("All V23 backend and text-interface tests passed.")

# Fault-analysis lifecycle: discovery pauses movement, locks one complete step,
# hides repair resources, and then resumes the paused Move command.
backend = DSRDashboardBackend(PROJECT, Path(tempfile.mkdtemp()))
backend.configure_max_time_step(20)
backend.load_faults_and_unknown(render_output=False)
backend.execute_current_step(
    mobile_commands=commands(
        backend,
        Scout={"type": "move", "waypoints": [[-4, 0]]},
    ),
    switch_operations=[],
    frames_per_step=4,
)
backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[],
    frames_per_step=4,
)
scout = backend.scouts["Scout"]
fault_b = backend.faults["B"]
assert fault_b.is_known(backend.event_id)
assert not fault_b.analyzed
assert scout.state == "Analyzing" and scout.analysis_locked
locked_position = scout.position
locked_path = list(scout.planned_path)
row_b = next(row for row in backend.fault_rows() if row[0] == "B")
assert row_b[4] == "Hidden" and row_b[5] == "Hidden"

# A locked object cannot accept a replacement command, and rejection is atomic.
locked_time = backend.time_step
try:
    backend.execute_current_step(
        mobile_commands=commands(
            backend,
            Scout={"type": "move", "waypoints": [[0, 0]]},
        ),
        switch_operations=[],
        frames_per_step=1,
    )
except ValueError as error:
    assert "locked" in str(error).lower()
else:
    raise AssertionError("Analysis-locked Scout accepted a Move command")
assert backend.time_step == locked_time
assert backend.scouts["Scout"].position == locked_position
assert backend.scouts["Scout"].planned_path == locked_path

# Continue represents the mandatory automatic analysis step. No movement occurs.
result = backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[],
    frames_per_step=4,
)
scout = backend.scouts["Scout"]
assert scout.position == locked_position
fault_b = backend.faults["B"]
assert fault_b.analyzed
assert not scout.analysis_locked
assert scout.state == "Moving"
assert scout.planned_path == locked_path
assert "Automatic fault analysis" in result["mobile_commands"]["Scout"]
row_b = next(row for row in backend.fault_rows() if row[0] == "B")
assert row_b[4] == str(fault_b.repair_resource)
assert row_b[5] == str(fault_b.remaining_repair_resource)

# A following Continue resumes the interrupted route.
backend.execute_current_step(
    mobile_commands=commands(backend),
    switch_operations=[],
    frames_per_step=4,
)
assert backend.scouts["Scout"].position != locked_position

# A known but unanalyzed fault triggers analysis when RC/Scout arrives exactly.
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
assert "status_tables" in downloaded_state
assert "faults" in downloaded_state["status_tables"]

html = (PROJECT / "index_v23.html").read_text(encoding="utf-8")
assert 'id="download-state"' in html
assert 'id="download-commands"' in html
assert "Download Pending Commands JSON" in html
assert "unit.analysis_locked" in html

print("All V23 fault-analysis and download tests passed.")


# One-click scenario reset is shared by the graphical and text interfaces.
async def reset_endpoint_tests():
    class ResetRequestStub:
        def __init__(self, payload):
            self.payload = payload

        async def json(self):
            return copy.deepcopy(self.payload)

    response = await web.load_state(ResetRequestStub({
        "mode": "reset",
        "max_time_step": 8,
    }))
    body = json.loads(response.body)
    assert response.status_code == 200 and body["ok"]
    assert web.backend.time_step == 0
    assert web.backend.game_score.max_time_step == 8
    assert len(web.backend.game_score.power_history) == 1

    response = await web.execute_text_command(ResetRequestStub({
        "switch_commands": {},
        "mobile_commands": {},
    }))
    assert response.status_code == 200
    assert web.backend.time_step == 1

    response = await web.load_state(ResetRequestStub({"mode": "reset"}))
    body = json.loads(response.body)
    assert response.status_code == 200 and body["ok"]
    assert web.backend.time_step == 0
    assert web.backend.game_score.max_time_step == 8
    assert len(web.backend.game_score.power_history) == 1
    assert body["message"].startswith(
        "Fault scenario reset to time step 0"
    )

asyncio.run(reset_endpoint_tests())

text_html = (PROJECT / "text_v23.html").read_text(encoding="utf-8")
assert 'id="reset-scenario"' in text_html
assert 'JSON.stringify({mode: "reset"})' in text_html
assert 'id="load-scenario"' in html
assert 'document.getElementById("load-scenario").disabled = false' in html

print("All V23 one-click scenario reset tests passed.")
