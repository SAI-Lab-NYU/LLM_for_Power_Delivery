# DSR Web v23 Test Report

## Automated tests

The included `test_v23.py` verifies the retained V22 functions and the new V23
behavior.

### Retained functions

- Cumulative weighted score calculation using `sum(abs(P) * W)` per step.
- Permanent score reset and lock when any previously energized load becomes de-energized.
- Maximum-time-step ending and command rejection after completion.
- Early completion and projected scoring.
- Multiple, overlapping, reversed, and out-of-map unknown rectangles.
- Signed active loads and negative transmission-grid active power.
- Atomic text-command validation and execution.

### V23-HF4 absolute-power weighted scoring tests

- The per-step contribution equals `sum(abs(P) * W)` over energized loads.
- Negative active-power loads contribute their absolute active-power magnitude.
- The graphical and text states expose weighted and unweighted load totals separately.
- A lower aggregate weighted contribution does not reset the score when all
  previously energized loads remain energized.
- Loss of any previously energized load resets the score to zero.
- The names of lost loads are reported in `violation_loads`.
- Negative-P loads do not cause a false monotonicity violation.
- Early-completion projection uses the fully restored weighted value.

### V23 fault-analysis tests

- PAC stops when it discovers an unknown fault.
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


### Fault-information-hiding regression tests

- A known but unanalyzed fault is drawn without numeric repair-resource values.
- The graphical hover text reports both required and remaining resources as
  `Hidden` until analysis completes.
- Numeric repair-resource values appear only after RC/PAC analysis completes.
- The JSON text state represents hidden resource fields as `null`.
- A completely unknown fault is omitted from the text-state fault list.
- A completely unknown fault is omitted from the Element States fault rows.
- Pixel-inspector payloads do not reveal an unknown fault identifier.
- `/api/text/state`, `/api/text/state.txt`, `/api/download/state.json`, and the
  predicted state returned by `/api/text/validate` preserve the same visibility
  rules.

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
All V23-HF4 backend and text-interface tests passed.
All V23 fault-analysis and download tests passed.
All V23 fault-information-hiding regression tests passed.
All V23-HF4 absolute-power weighted scoring regression tests passed.
```

## HTTP smoke test

The FastAPI application was started locally and `/api/state` returned the V23
base-map state. The returned Object metadata reported analysis enabled for RC1,
RC2, and PAC1, and disabled for MPS1.

### V23-HF4 negative-active-power verification

- A test load with `P = -175 kW` and `W = 5` contributes `875` points.
- Its signed restored active-power value remains `-175 kW` in the physical state.
- Its weighted score value is `abs(-175) * 5 = 875`.
- The graphical summary and text state report the same absolute-power weighted value.
