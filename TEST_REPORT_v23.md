# DSR Web v23 Test Report

## Automated tests

The included `test_v23.py` verifies the retained V22 functions and the new V23
behavior.

### Retained functions

- Cumulative score calculation.
- Permanent score reset and lock after restored active power decreases.
- Maximum-time-step ending and command rejection after completion.
- Early completion and projected scoring.
- Multiple, overlapping, reversed, and out-of-map unknown rectangles.
- Signed active loads and negative transmission-grid active power.
- Atomic text-command validation and execution.

### V23 fault-analysis tests

- Scout stops when it discovers an unknown fault.
- The fault becomes known but remains unanalyzed.
- Required and remaining repair resources display as `Hidden`.
- The discovering unit is locked for one complete step.
- A replacement Move command is rejected atomically during the lock.
- `Continue` performs automatic analysis without movement.
- The fault resources become visible after the analysis step.
- The interrupted Move route is preserved and resumes later.
- An RC arriving at a known but unanalyzed fault must analyze it before repair.
- MPS has the inherited analysis interface disabled, continues moving after
  discovery, and leaves the fault unanalyzed.


### One-click reset tests

- The text page exposes a **Reset Fault Scenario** button.
- The graphical reset button remains enabled after game completion.
- Resetting from an active game returns the simulator to time step `0`.
- Score history is cleared and restarted with exactly one `P0` entry.
- Mobile objects, sources, faults, resources, and topology are rebuilt.
- The configured maximum time step `N` is preserved when the reset request
  does not provide a replacement.

### Download tests

- `/api/download/state.json` is registered.
- The response is a JSON attachment.
- The downloaded state includes the game score and all status-table payloads.
- The graphical page contains both state and pending-command download buttons.

Run:

```powershell
python test_v23.py
```

Expected final output:

```text
All V23 backend and text-interface tests passed.
All V23 fault-analysis and download tests passed.
```

## HTTP smoke test

The FastAPI application was started locally and `/api/state` returned the V23
base-map state. The returned Object metadata reported analysis enabled for RC1,
RC2, and Scout, and disabled for MPS1.
