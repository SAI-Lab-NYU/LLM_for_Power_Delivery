# DSR Grid Map V23-HF4

V23 is a local FastAPI + Plotly distribution-system-restoration simulator. It
includes the graphical dashboard and an additional JSON text-command
interface.

## Main V23 changes

### Game score

Set the maximum completed time step `N` before loading the fault scenario.
For each completed time step `t`, the scoring contribution is:

```text
S_t = sum(P_j * W_j) for every energized load j
```

The displayed cumulative score after time step `i` is:

```text
Score_i = S_0 + S_1 + ... + S_i
```

The score is no longer invalidated by comparing the aggregate value at two
adjacent steps. This is important because active power may be negative.
Instead, the simulator records the energized state of every load-bearing Node.
Once a load is energized at a completed time step, it must remain energized at
every later completed time step. If any previously energized load becomes
de-energized, the score becomes zero and remains permanently locked at zero.

The game ends when the completed time step reaches `N`. It also ends early
when all ordinary and transmission-source faults are cleared and every load
Node is energized. For an early completion at step `k < N`, the remaining
`N-k` steps are scored using the fully restored weighted value
`sum(P_j * W_j)`.

After a game ends, command execution and Switch validation are disabled. Load
the base map before loading a new fault scenario.

### Multiple unknown rectangles

The scenario JSON file and text/API interface accept zero or more rectangles.
Their union defines the initially unknown Pixels. The graphical command panel
does not expose unknown-region editing.

- Rectangles may overlap.
- Corner order may be reversed; it is normalized automatically.
- Rectangles may extend outside the map. Only map Pixels are affected.
- An empty rectangle list means that every non-protected Pixel starts known.

### Signed active load

Node active load may be positive or negative. When the transmission grid is
connected, signed grid power is supported; a negative grid P value represents
net power absorption by the transmission grid.

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
    "PAC1": {"type": "stay"},
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

## Run locally

From this directory:

```powershell
python -m pip install -r requirements.txt
python simulator_web_v23.py
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

## Hotfix HF1: hidden fault information

Before RC/PAC analysis, both required and remaining repair resources are always
displayed as `Hidden` and exported as `null`. Completely unknown faults are omitted
from text state, Element States, and Pixel Inspector payloads.

After replacing the project, stop the old server with `Ctrl+C` and run
`python simulator_web_v23.py` again. A running Python process does not reload an
already imported Plotly module. HF1 introduced the explicit build identifier. The current page title and
`/api/state` now show `V23-HF4` so the active build can be verified.

## Hotfix HF2: weighted scoring and per-load continuity

- Each time-step contribution was changed from plain restored P to a load-priority weighted value.
- Score reset is based on the loss of any previously energized load, not on an aggregate power decrease.
- State JSON includes `weighted_load_history`, `energized_load_history`,
  `violation_loads`, `weighted_restored_load`, and `total_weighted_load`.

## Hotfix HF3: absolute-power weighted scoring

- Each time-step contribution is `sum(abs(P) * W)` over currently energized loads.
- A negative-P load contributes its active-power magnitude rather than a negative score.
- Physical power-balance values and `power_history` remain signed.
- Early-completion projection uses the full-system `sum(abs(P) * W)` value.
- The graphical score panel displays both signed restored P and score-weighted `|P| x W`.

## Hotfix HF4: Patrol and Assessment Crew terminology

The former inspection-only object is now named **Patrol and Assessment Crew (PAC)**.
Its default object name is `PAC1`, its displayed type is `PAC`, and its configuration
is stored in `PatrolAssessmentCrews_v1.json`. All simulator behavior is unchanged.
