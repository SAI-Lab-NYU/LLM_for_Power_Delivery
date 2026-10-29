# IEEE 33-Bus DSR Simulator (V23)

This folder contains a runnable **simulator**, not an optimizer or a saved
restoration replay. It uses the DSR Grid Map V23 interface and engine with an
IEEE 33-bus electrical case. The Basic13 case in the repository root is
unchanged and has its own launcher.

## Quick Start

From the repository root, install Python 3 and the shared dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
./run_ieee33.sh
```

On Windows PowerShell, use `.venv\Scripts\python -m pip install -r
requirements.txt`, then `./run_ieee33.ps1`. The 33-bus dashboard is served at
`http://127.0.0.1:8067`. To run without automatically opening a browser, use
`./run_ieee33.sh --no-browser`. The Basic13 launcher uses port 8050.

On the dashboard, **Load Base Map** shows the fault-free starting network.
**Reset Fault Scenario** loads the three configured physical faults and unknown
regions. Set switch and unit commands at right, validate if needed, then
**Execute Current Step**. A separate text-command interface is at `/text`.
The state and event endpoints include `/api/state`, `/api/text/state`, and
`/api/events?after=0`. The command API is documented by the root repository's
V23 command reference; this branch does not bundle a controller.

## Case Data

| Item | Configuration |
| --- | --- |
| Electrical network | 33 buses, 32 normally radial branches, 5 tie lines |
| Design partition | 8 static zones: Z1, Z2A, Z2B, Z3, Z4A, Z4B, Z5A, Z5B |
| Switches | 13, including remote and manual switches |
| Field resources | RC1, RC2, one inspection crew (Scott1 in the engine), MPS1 |
| Configured faults | F13_14, F24_25, F30_31 |
| Main source | B1, available from the start |
| Total nominal demand | 3,715 kW / 2,300 kvar |
| Simulation clock | 5 min per command step, with four internal physics frames |

The electrical topology, bus loads, and branch impedances are based on the
standard IEEE 33-bus benchmark. **Bus coordinates, roads, zone boundaries,
fault positions, unknown regions, crews, and MPS are project assumptions**, not
official IEEE GIS data. The synthetic line display length is mapped from
impedance magnitude to approximately 2-20 Pixels. The single-line display
uses compact orthogonal tie-line routes; these are drawing paths, not extra
electrical branches or roads. `scenario.json` records the zone membership,
load totals, branch impedances, and provenance.

The engine's `A1`-`A8` names are backend Areas. The `Z*` names in
`scenario.json` are the design zones; see `backend_area_by_design_zone` for
their one-to-one mapping. A tie-line operation can change electrical supply
without changing these static patrol partitions.

## Files

- `ieee33_impedance_simulator_web.py`: case-specific FastAPI entry point and
  map display adjustments.
- `simulator_core_v23.py`, `simulator_plotly_v23.py`, `simulator_web_v23.py`:
  V23 simulation, visualization, and HTTP API engine.
- `index_v23.html`, `text_v23.html`: graphical and text interfaces.
- `Nodes_v5.json`, `ElectricalLines_v4.json`, `Switch_v5.json`, `Roads_v1.json`
  and other `*_v*.json`: fixed, versioned scenario inputs.
- `test_ieee33_package.py`: package smoke and scenario-consistency checks.
- `offline_demo/IEEE33_Standalone_Simulator_Demo.html`: optional single-file,
  server-free teaching demo. It is **not** the V23 physics engine; use the
  Python app for experiments or API integration.

Generated files are written under `ieee33/runtime/` and are Git-ignored.
No Gurobi plan, result trajectory, or benchmark output is included.

## Modeling Scope

This is a topology-oriented restoration simulator. It tracks field movement,
inspection knowledge, fault analysis and repair, switch states, mobile supply,
and load energization. It does **not** solve an optimization problem or run
OpenDSS/three-phase power flow. The road network is synthetic and not a GIS
measurement; travel and patrol outcomes should be interpreted in that light.
