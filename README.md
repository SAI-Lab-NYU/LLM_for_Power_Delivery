# DSR Grid Map v23

V23 is a local FastAPI + Plotly distribution-system-restoration simulator. It
includes the graphical dashboard and an additional JSON text-command
interface.


## Overview

This package is the original standalone V23 simulator for a synthetic
**Basic13 distribution-system-restoration (DSR) case**. It is a teaching and
prototyping system rather than a reproduction of the IEEE 13-node test feeder.
The dashboard lets a user inspect uncertain regions, analyze and repair
faults, operate switches, dispatch a mobile power source, and observe load
restoration.

The base case contains:

| Item | Base-case contents |
|---|---|
| Electrical network | 13 primary Nodes (`N1`-`N13`) and 4 auxiliary tie Nodes (`V1`-`V4`) |
| Network partition | 5 static Areas derived from topology and Switch boundaries |
| Lines and Switches | 19 ElectricalLines and 8 Switches |
| Damage | 2 fixed physical faults (`A` and `B`) |
| Field units | 2 repair crews (`RC1`, `RC2`) and 1 inspection Scout (`Scout`) |
| Mobile supply | 1 mobile power source (`MPS1`) |
| Main source | Transmission Grid at `N1`, with a configured restoration delay |
| Loads | Active power, reactive power, and priority weight stored on load Nodes |

A run begins from the physical state defined by the JSON files. Some map
Pixels are hidden by `UnknownState_v1.json`, while the simulator retains the
complete physical state internally. The user issues one atomic command batch
per time step. Restoration ends when the configured horizon is reached or all
faults are cleared and every load Node is energized.

## Restoration workflow and execution semantics

The original V23 model uses **abstract simulation steps**, not calibrated
minutes. Each accepted command batch advances one complete time step. The
step is divided into four animation frames so movement and repair can be shown
gradually; these frames are not additional decision epochs.

- All unit commands and Switch commands in a batch are validated together. If
  any command is invalid, the complete batch is rejected and the live state is
  unchanged.
- Omitting a mobile unit applies `Continue`, preserving an eligible unfinished
  movement, analysis, repair, or supply action.
- `move_speed` is the Level-1-road travel rate in Pixels per step. Level-2 roads
  use a 1.5 speed multiplier. Off-road movement is allowed at one Pixel per
  step.
- Every Node and ElectricalLine Pixel is registered as a Level-1 service road.
  Explicit roads may provide Level-2 travel.
- Inspection uses a square Chebyshev neighborhood around the unit. For
  example, range 2 covers every Pixel whose horizontal and vertical offsets
  from the unit are both at most 2.
- `repair_speed` is measured in repair-resource units per step. A repair crew
  must be at the fault Pixel, and the fault must have completed analysis.
- A newly discovered fault requires one complete automatic analysis step
  before its repair-resource requirement becomes visible.
- In this original release, Switch commands take effect within the command
  step; remote and manual operation times are not represented separately.
- Energization and the game score are recomputed after the step finishes.

## Information states

The simulator separates the hidden physical state from what field units have
inspected. Unknown does not mean healthy.

```text
Unknown Pixel
    -> inspected by a field unit
Known fault discovered
    -> one automatic analysis step by RC or Scout
Analyzed fault; repair requirement visible
    -> repair resource applied by RC
Repaired; historical inspection knowledge retained
```

At scenario initialization, Pixels outside the configured unknown rectangles
are marked inspected. A unit reveals Pixels as it moves or operates within its
inspection range. `RC1`, `RC2`, and `Scout` can discover faults; a discovering
RC or Scout pauses for automatic analysis. `MPS1` can reveal nearby Pixels but
does not analyze faults. Fault repair requirements remain hidden until analysis
completes.

The simulator maintains three distinct concepts:

1. **Physical truth:** actual faults, physical Switch states, stored energy,
   remaining repair work, and source availability.
2. **Inspection knowledge:** whether each Pixel and its associated equipment
   have been observed in the current event.
3. **Operational state:** current unit locations and actions, analyzed repairs,
   Switch configuration, energized Nodes, and the score.

The text interface is intended to expose an operational view, while the
backend retains complete truth for simulation and rendering.

## Package layout

| File | Purpose |
|---|---|
| `GRAPH.json` | Integer Pixel-map bounds and map name |
| `Nodes_v5.json` | Node positions, types, loads, and load-priority weights |
| `ElectricalLines_v4.json` | ElectricalLine endpoints and complete Pixel routes |
| `Switch_v5.json` | Switch placement, initial state, and controlled terminal |
| `Roads_v1.json` | Explicit road polylines and road levels |
| `Faults_v1.json` | Fixed ground-truth fault locations and repair requirements |
| `UnknownState_v1.json` | Initial unknown rectangles and conservative Switch opening rules |
| `RepairCrews_v1.json` | Repair-crew locations, movement, inspection, repair, and resource parameters |
| `Scout_v1.json` | Scout location, movement speed, and inspection range |
| `MPS_v1.json` | Mobile-source location, limits, movement, inspection, and legacy energy parameter |
| `TransmissionSource_v1.json` | Main-source Node and restoration delay |
| `simulator_core_v23.py` | Map loading, knowledge state, movement, inspection, repair, switching, energization, and scoring |
| `simulator_plotly_v23.py` | Plotly map and state visualization |
| `simulator_web_v23.py` | FastAPI application, graphical endpoints, and text-command API |
| `index_v23.html` | Graphical dashboard |
| `text_v23.html` | JSON state and command page |
| `test_v23.py` | Original backend, interface, analysis, reset, and download regression tests |
| `DSR_V23_Command_Reference.md` | Detailed command and field reference |

## Main V23 changes

### Game score

Set the maximum completed time step `N` before loading the fault scenario.
When the scenario is loaded, the simulator records restored active power
`P0` and initializes the score with `P0`. After completed time step `i`, the
score is:

```text
P0 + P1 + ... + Pi
```

If `Pi < P(i-1)`, the score becomes zero and remains permanently locked at
zero for the rest of that game. The simulator may continue running, but later
commands cannot increase the score.

The game ends when the completed time step reaches `N`. It also ends early
when all ordinary and transmission-source faults are cleared and every load
Node is energized. For an early completion at step `k < N`, the remaining
`N-k` steps are scored at the fully restored active power.

After a game ends, command execution and Switch validation are disabled. The
**Reset Fault Scenario** action remains available and rebuilds the complete
scenario directly at time step 0.

### Multiple unknown rectangles

The `UnknownState_v1.json` file accepts zero or more rectangles. Their union
defines the initially unknown Pixels. The graphical command column does not
show unknown-region editors.

- Rectangles may overlap.
- Corner order may be reversed; it is normalized automatically.
- Rectangles may extend outside the map. Only map Pixels are affected.
- An empty rectangle list means that every non-protected Pixel starts known.

### Signed active load

Node active load may be positive or negative. When the transmission grid is
connected, signed grid power is supported; a negative grid P value represents
net power absorption by the transmission grid.


### One-click scenario reset

Both the graphical dashboard and the text-command page include a
**Reset Fault Scenario** button. One click performs a complete scenario
restart:

- time step returns to `0`;
- score history restarts from the new `P0`;
- faults and unknown Pixels reload from the scenario files;
- switches, sources, mobile-unit positions, routes, resources, and MPS energy
  return to their initial values;
- the currently configured maximum time step `N` is preserved unless the
  graphical request supplies a new value.

The same behavior is available through:

```text
POST /api/load
{"mode": "reset"}
```

### Text-command interface

Open:

```text
http://127.0.0.1:8050/text
```

The text page displays the current state as JSON and accepts one complete
command batch in this form:

```json
{
  "switch_commands": {
    "S1": "opened",
    "S2": "closed"
  },
  "mobile_commands": {
    "RC1": {"type": "continue"},
    "RC2": {"type": "move", "waypoints": [[0, 0]]},
    "Scout": {"type": "stay"},
    "MPS1": {"type": "supply", "p": 800, "q": 440}
  }
}
```

Omit a Switch to preserve its current state. Omit an Object command to apply
`Continue`.

Available endpoints:

```text
GET  /api/text/state
GET  /api/text/state.txt
POST /api/text/validate
POST /api/text/execute
```

`/api/text/validate` predicts the next state on a copy and never changes the
live time step. `/api/text/execute` uses the same atomic validation as the
web dashboard. A rejected command returns an error and leaves the simulator
at the current time step.


## Known limitations

This original simulator intentionally uses a compact, synthetic model.
Results should be interpreted within the following boundaries:

- Basic13 is not the IEEE 13-node test feeder, and its integer map coordinates
  are not measured GIS distances.
- Load energization is based on topology, fault status, source connection, and
  MPS limits. The simulator does not solve AC power flow or enforce voltage,
  current, thermal, or protection constraints.
- Time is measured in abstract steps. The model does not attach a physical
  number of minutes to movement, analysis, repair, or source restoration.
- Faults are fixed Pixel-level objects loaded from `Faults_v1.json`. There is no
  stochastic weather field, component fragility model, line-length effect, or
  multiple independent damage segments on one line.
- Every electrical corridor is available as a Level-1 service road. This is a
  reachability assumption, not a surveyed road-access model.
- Switch actions are instantaneous within a step; manual travel and operation
  time are not modeled.
- The original MPS `energy` parameter uses legacy per-step accounting and
  should not be interpreted directly as calibrated kWh.
- The graphical and API payloads were designed for teaching and debugging.
  They are not a formally audited partially observable controller contract;
  an external controller should use an explicit visible-field whitelist.
- The package executes user or API commands but does not include an optimal
  restoration policy, rolling-horizon optimizer, or posterior risk update.

## Run locally

From this directory:

On macOS or Linux:

```bash
python3 -m pip install -r requirements.txt
./run_v23.sh
```

On Windows PowerShell:

```powershell
python -m pip install -r requirements.txt
.\run_v23.ps1
```

The graphical dashboard opens at:

```text
http://127.0.0.1:8050
```

You may also use `run_v23.bat`, `run_v23.ps1`, or `run_v23.sh`.

## Graphical unknown-area configuration

The graphical command column does not expose unknown-region editors. The regular
**Reset Fault Scenario** action loads unknown rectangles from `UnknownState_v1.json`.
The text/API interface can still supply an `unknown_regions` list when a custom
scenario is required.

The default file now contains the original unknown rectangle plus a 3-by-3 Pixel
square adjoining node V4: Pixel centers `x = 10..12`, `y = -18..-16`. V4 is the upper-left Pixel of this square, so the initial RC2 inspection range does not immediately clear it.
