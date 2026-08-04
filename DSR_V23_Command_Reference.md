# DSR Simulator V23 Command Reference

## 1. Purpose

This document defines the environment-control commands, mobile-object commands, Switch commands, request formats, validation workflow, and common examples for DSR Simulator V23.

The simulator server normally runs at:

```text
http://127.0.0.1:8050
```

The graphical interface and the text interface use the same live simulator backend. A command executed from either interface changes the same environment state.

---

## 2. Recommended Game-Start Command

A new game can be started with **one HTTP request**. It is not necessary to load the Base Map first and then load the fault scenario separately.

### Request

```http
POST /api/load
Content-Type: application/json
```

```json
{
  "mode": "reset",
  "max_time_step": 20
}
```

### Meaning

This request performs a complete restart:

1. Reloads the Base Map and all configured objects.
2. Reloads faults from the scenario files.
3. Reloads the default unknown regions from `UnknownState_v1.json`.
4. Resets all RCs, Patrol and Assessment Crew (PAC) units, and MPS units to their initial states.
5. Restores initial Switch states and the configured transmission-source outage.
6. Resets the simulation to `Time Step = 0`.
7. Resets the score and immediately records the T0 weighted contribution `sum(abs(P) * W)` over energized loads.
8. Sets the maximum completed time step to `N = 20` in this example.

The aliases `"scenario"` and `"faults"` currently have the same effect as `"reset"`, but `"reset"` is the recommended name for starting or restarting a game.

---

## 3. Environment-Control Commands

Environment-control commands initialize, inspect, validate, execute, or stop the simulator. They are separate from the mobile-object command batch used to advance one time step.

### 3.1 Load the Base Map Only

```http
POST /api/load
Content-Type: application/json
```

```json
{
  "mode": "base"
}
```

Optional configuration:

```json
{
  "mode": "base",
  "max_time_step": 30
}
```

**Meaning:** Reloads the map, network elements, Switches, roads, and mobile objects without loading the fault scenario. The simulator returns to `Time Step = 0`, but the game has not started and no weighted T0 contribution has been scored.

**Typical use:** Inspecting the clean Base Map or changing `N` before starting a scenario.

---

### 3.2 Reset and Load the Fault Scenario

```http
POST /api/load
Content-Type: application/json
```

```json
{
  "mode": "reset"
}
```

Optional one-request game configuration:

```json
{
  "mode": "reset",
  "max_time_step": 20
}
```

**Meaning:** Performs a complete restart and loads the file-defined fault and unknown-region scenario. The current configured `N` is preserved unless `max_time_step` is included.

---

### 3.3 Reset with Text-Defined Unknown Regions

The graphical command panel does not edit unknown regions. A text/API request may override the file-defined rectangles:

```http
POST /api/load
Content-Type: application/json
```

```json
{
  "mode": "reset",
  "max_time_step": 20,
  "unknown_regions": [
    {
      "name": "Region1",
      "lower_left": [-13, -17],
      "upper_right": [3, 3]
    },
    {
      "name": "V4UnknownSquare3x3",
      "lower_left": [10, -18],
      "upper_right": [12, -16]
    }
  ]
}
```

Rules:

- Multiple rectangles are combined by union.
- Rectangles may overlap.
- Corner order may be reversed; the simulator normalizes it.
- Rectangles may extend outside the map; only in-map Pixels become unknown.
- Omitting `unknown_regions` loads the rectangles from `UnknownState_v1.json`.

---

### 3.4 Configure the Maximum Time Step

```http
POST /api/game/configure
Content-Type: application/json
```

```json
{
  "max_time_step": 20
}
```

**Meaning:** Sets the maximum completed time step `N`.

Rules:

- `N` must be a positive integer.
- `N` should be configured before the fault scenario starts.
- After a game starts, `N` cannot be changed to a different value. Reload the Base Map or use a new Reset request with the desired `max_time_step`.

---

### 3.5 Read the Current State

#### Text-oriented JSON state

```http
GET /api/text/state
```

#### Text-formatted JSON

```http
GET /api/text/state.txt
```

#### Graphical-page state

```http
GET /api/state
```

#### Downloadable environment output

```http
GET /api/download/state.json
```

The downloaded state contains the current time step, score, objects, faults, Switches, loads, power clusters, unknown regions, and the expected command format.

---

### 3.6 Validate Only the Pending Switch Plan

```http
POST /api/validate-switches
Content-Type: application/json
```

```json
{
  "switch_commands": {
    "S2": "opened",
    "S4": "closed"
  }
}
```

**Meaning:** Tests the provisional final topology without changing the live state. The response includes MPS supply contexts calculated from the candidate topology.

This validation does not:

- change a Switch;
- advance time;
- move an object;
- consume energy or repair resources.

Use exactly `"opened"` or `"closed"` in API requests.

---

### 3.7 Validate a Complete One-Step Command Batch

```http
POST /api/text/validate
Content-Type: application/json
```

```json
{
  "switch_commands": {
    "S2": "opened"
  },
  "mobile_commands": {
    "RC1": {"type": "continue"},
    "RC2": {"type": "move", "waypoints": [[0, 0]]},
    "PAC1": {"type": "stay"},
    "MPS1": {"type": "supply", "p": 300, "q": 140}
  }
}
```

**Meaning:** Predicts one complete time step on a copied simulator. The live state is not changed.

A successful response contains:

```json
{
  "ok": true,
  "message": "Command batch is executable.",
  "live_time_step": 0,
  "predicted_state": {}
}
```

A rejected command returns `"ok": false`, an error message, and the unchanged live time step.

---

### 3.8 Execute One Time Step

#### Text interface

```http
POST /api/text/execute
Content-Type: application/json
```

#### Graphical interface backend

```http
POST /api/execute
Content-Type: application/json
```

Both endpoints accept the same command body. `/api/execute` additionally returns browser animation frames. `/api/text/execute` returns the resulting text state without the four-frame graphical animation.

Execution is atomic: if any command or constraint is illegal, the complete batch is rejected and the live state remains unchanged.

---

### 3.9 Stop the Local Server

```http
POST /api/shutdown
Content-Type: application/json
```

```json
{}
```

---

## 4. One-Step Command-Batch Format

Every normal simulation step uses this top-level JSON structure:

```json
{
  "switch_commands": {},
  "mobile_commands": {}
}
```

A complete example is:

```json
{
  "switch_commands": {
    "S2": "opened"
  },
  "mobile_commands": {
    "RC1": {
      "type": "continue"
    },
    "RC2": {
      "type": "move",
      "waypoints": [[0, 5], [0, 0]]
    },
    "PAC1": {
      "type": "stay"
    },
    "MPS1": {
      "type": "supply",
      "p": 300,
      "q": 140
    }
  }
}
```

Notes:

- `switch_commands` must be a JSON object.
- `mobile_commands` must be a JSON object.
- An omitted mobile object defaults to `Continue`.
- An empty `switch_commands` object means that no Switch is operated.
- Switch actions are evaluated before mobile-object commands.

---

## 5. Switch Commands

### Format

```json
{
  "switch_commands": {
    "S1": "opened",
    "S8": "closed"
  }
}
```

### Meaning

- `"opened"`: physically opens the Switch.
- `"closed"`: physically closes the Switch.

### Main Constraints

- The Switch name must exist.
- An unknown Switch cannot be operated.
- The final simultaneous Switch topology must remain radial; an energized loop is illegal.
- Closing from an energized or newly energized side cannot connect an unknown Area or an Area with an active fault.
- Before the transmission source restores, the source-connected network must satisfy the restoration-safety checks.
- Multiple Switch operations are evaluated as one simultaneous final topology.

Example: a closing operation that would create a loop may become legal when another Switch is opened in the same batch.

---

## 6. Mobile-Object Command Summary

| Command | RC | PAC | MPS | Meaning |
|---|---:|---:|---:|---|
| `continue` | Yes | Yes | Yes | Continue the current valid activity or remain Idle. |
| `move` | Yes | Yes | Yes | Replace the current activity with movement through ordered waypoints. |
| `stay` | Yes | Yes | Yes | Cancel movement or other active operation and become Idle. MPS disconnects. |
| `repair` / `repaire` | Yes | No | No | Apply a specified repair-resource amount during the next step. |
| `supply` | No | No | Yes | Connect at the current Node and output specified P and Q. |

The API accepts both `"repair"` and the legacy spelling `"repaire"`.

---

## 7. Continue Command

### Format

```json
{
  "type": "continue"
}
```

The aliases `"keep"` and an empty command type are also interpreted as Continue, but `"continue"` is recommended.

### Behavior by Object State

- **Moving RC or PAC:** continues the remaining planned path.
- **Moving MPS:** continues the remaining planned path.
- **Repairing RC:** repeats the previous repair amount when the same repair is still legal; otherwise the RC becomes Idle.
- **Supplying MPS:** continues the current P/Q output.
- **Idle object:** remains Idle.
- **Analysis-locked RC or PAC:** performs the mandatory automatic analysis step. No alternative command is accepted during that step.

---

## 8. Move Command

### One Destination

```json
{
  "type": "move",
  "waypoints": [[0, 0]]
}
```

### Ordered Intermediate Waypoints and Final Destination

```json
{
  "type": "move",
  "waypoints": [[5, 0], [5, -8], [-4, -8]]
}
```

### Rules

- A Move command requires at least one waypoint.
- A Move command accepts at most five waypoints.
- Every waypoint must contain exactly two integer coordinates.
- Every waypoint must be inside the map.
- Waypoints are mandatory and are visited in the listed order.
- The simulator selects a minimum-time path between consecutive waypoints using road-speed information.
- Movement is orthogonal between neighboring Pixels.
- A new Move command replaces any previous movement, repair, or supply command.
- An MPS disconnects before starting to move.

### Fault-Discovery Behavior

- RC and PAC have automatic fault analysis enabled.
- When an RC or PAC discovers a previously undiscovered fault during movement, it stops and preserves its remaining route.
- During the next complete time step, that object is locked for automatic fault analysis and cannot accept a different command.
- The fault location is visible immediately, but its repair resource remains hidden until analysis finishes.
- An RC or PAC arriving at an already discovered but unanalyzed fault also enters the one-step analysis lock.
- MPS has the analysis interface but it is disabled in V23. An MPS does not stop or analyze when it discovers a fault.

---

## 9. Stay Command

### Format

```json
{
  "type": "stay"
}
```

### Meaning

- Cancels the current movement route.
- Cancels an RC repair command.
- Disconnects an MPS that is supplying.
- Sets the object state to Idle.

A Stay command is rejected when an RC or PAC is locked for automatic fault analysis.

---

## 10. Repair Command

### Format

```json
{
  "type": "repair",
  "fault": "A",
  "amount": 5
}
```

Legacy spelling:

```json
{
  "type": "repaire",
  "fault": "A",
  "amount": 5
}
```

### Rules

A repair command is legal only when:

- the object is an RC;
- the fault exists and is active;
- the fault is known;
- the fault has completed the required analysis step;
- the RC is located on the exact fault Pixel;
- the RC is not analysis-locked;
- the RC has remaining repair resources;
- `amount` is a positive integer;
- `amount` does not exceed the smallest of:
  - the RC repair speed;
  - the RC remaining resources;
  - the fault remaining repair resource.

After a partial repair, `Continue` repeats the previous repair amount when possible. If the fault is complete or the previous amount is no longer legal, the RC becomes Idle.

---

## 11. MPS Supply Command

### Format

```json
{
  "type": "supply",
  "p": 800,
  "q": 440
}
```

Units:

- `p`: kW
- `q`: kVar

### MPS-Level Constraints

- `p` and `q` must be numeric.
- `p` must be nonnegative.
- `p <= P_limit`.
- `sqrt(p^2 + q^2) <= S_limit`.
- The MPS must be located exactly at a Node.
- The connection Node must be known and must not have a direct active fault.
- The MPS must have enough stored energy to support one complete step. Otherwise, it is automatically disconnected before the step.

### Cluster Power-Balance Rule

For a cluster without the transmission grid:

```text
sum(P_MPS) = total active load of the connected Areas
sum(Q_MPS) = total reactive load of the connected Areas
```

For a cluster connected to the transmission grid:

```text
P_grid = P_load - sum(P_MPS)
Q_grid = Q_load - sum(Q_MPS)
```

The transmission grid automatically supplies the remaining power. The MPS active-power total cannot force the grid to absorb an unsupported active-power surplus under the current rules.

### Energy Consumption

For a one-step output of `p` kW:

```text
Energy(next) = Energy(current) - p kW.step
```

Reactive power does not reduce the stored energy in the current model.

---

## 12. Automatic Fault-Analysis Lock

When an RC or PAC begins analyzing a fault, the object is locked for one complete time step.

Recommended command during the locked step:

```json
{
  "type": "continue"
}
```

It is also legal to omit that object from `mobile_commands`, because omission defaults to Continue.

The following commands are illegal during the lock:

- Move
- Stay
- Repair
- Any other replacement command

A rejected command does not advance time and does not partially apply Switch or other object commands.

---

## 13. Command Processing Order

For each successful time step, V23 processes the batch in this order:

1. Copy the live environment for atomic validation.
2. Apply all requested Switch targets simultaneously.
3. Validate Switch safety, source-restoration safety, and radiality.
4. Detect automatic analysis that must begin because an RC or PAC is already at a known, unanalyzed fault.
5. Validate and apply RC, PAC, and MPS commands.
6. Check MPS energy and automatically disconnect an MPS that cannot sustain one full step.
7. Validate cluster P/Q balance.
8. Advance all mobile objects for one complete time step.
9. Update fault discovery, analysis, repair, MPS energy, network states, loads, and score.
10. Commit the copied state to the live environment only if every check succeeds.

---

## 14. Common Complete Command Examples

### 14.1 Advance One Step Without Replacing Any Current Command

```json
{
  "switch_commands": {},
  "mobile_commands": {}
}
```

Because omitted objects default to Continue, this advances all objects using their current commands.

An explicit version is:

```json
{
  "switch_commands": {},
  "mobile_commands": {
    "RC1": {"type": "continue"},
    "RC2": {"type": "continue"},
    "PAC1": {"type": "continue"},
    "MPS1": {"type": "continue"}
  }
}
```

---

### 14.2 Move RC1 and PAC1

```json
{
  "switch_commands": {},
  "mobile_commands": {
    "RC1": {
      "type": "move",
      "waypoints": [[0, 0]]
    },
    "PAC1": {
      "type": "move",
      "waypoints": [[-4, 0], [-4, -8]]
    }
  }
}
```

RC2 and MPS1 are omitted and therefore receive Continue.

---

### 14.3 Repair Fault A

```json
{
  "switch_commands": {},
  "mobile_commands": {
    "RC1": {
      "type": "repair",
      "fault": "A",
      "amount": 5
    }
  }
}
```

This requires RC1 to be at Fault A and the fault to have completed analysis.

---

### 14.4 Open S2, Then Supply the Resulting Island

```json
{
  "switch_commands": {
    "S2": "opened"
  },
  "mobile_commands": {
    "MPS1": {
      "type": "supply",
      "p": 300,
      "q": 140
    }
  }
}
```

The Switch operation is applied first. The MPS balance is then checked against the cluster created by the new topology.

---

### 14.5 Simultaneous Topology Transfer

```json
{
  "switch_commands": {
    "S8": "closed",
    "S4": "opened"
  },
  "mobile_commands": {}
}
```

Both Switch targets are evaluated as one final topology. This can be legal even when closing S8 alone would create an energized loop.

---

## 15. Error Response and Atomic Rejection

A typical rejected request is:

```json
{
  "ok": false,
  "message": "Description of the failed constraint.",
  "time_step": 4,
  "game": {}
}
```

When validation or execution fails:

- the live time step does not change;
- Switch states do not partially change;
- mobile objects do not partially move;
- MPS energy is not consumed;
- RC repair resources are not consumed;
- the score does not change.

Use `/api/text/validate` before `/api/text/execute` when an external policy or agent should test a command without changing the environment.

---

## 16. Game-End Rules

The game ends when either condition is met:

1. The completed time step reaches `N`.
2. All faults are cleared and all load-bearing Nodes are energized before `N`.

For each completed time step `t`, the scoring contribution is `sum(abs(P) * W)` over all energized load-bearing Nodes. The cumulative score is the sum of these weighted contributions from T0 through the current step. Once a load has been energized at a completed time step, that load must remain energized at every later completed time step. If any previously energized load becomes de-energized, the score is reset to zero and remains permanently locked. The rule does not compare aggregate power values, so energizing a negative-P load contributes its absolute active-power magnitude and does not by itself trigger a score reset.

For early completion at time step `k < N`, the simulator projects the fully restored weighted value `sum(abs(P) * W)` through the remaining steps and adds it to the final score.

After the game ends, normal time-step commands are disabled. Start a new game with:

```json
{
  "mode": "reset"
}
```

or:

```json
{
  "mode": "reset",
  "max_time_step": 20
}
```

---

## 17. PowerShell Examples

The following examples assume that the server is already running.

### Start a New Game

```powershell
$baseUrl = "http://127.0.0.1:8050"
$body = @{
    mode = "reset"
    max_time_step = 20
} | ConvertTo-Json -Depth 10

Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/load" `
    -ContentType "application/json" `
    -Body $body
```

### Read the Current Text State

```powershell
Invoke-RestMethod -Method Get -Uri "$baseUrl/api/text/state"
```

### Validate a Command File

Assume `command.json` contains a complete command batch:

```powershell
$command = Get-Content .\command.json -Raw

Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/text/validate" `
    -ContentType "application/json" `
    -Body $command
```

### Execute the Same Command File

```powershell
Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/api/text/execute" `
    -ContentType "application/json" `
    -Body $command
```

### Save the Current State to a Local JSON File

```powershell
Invoke-WebRequest `
    -Uri "$baseUrl/api/download/state.json" `
    -OutFile .\current_state.json
```

---

## 18. Minimal External-Agent Workflow

An external controller can use the simulator as an environment with the following loop:

1. Start the game:

```text
POST /api/load
```

```json
{"mode": "reset", "max_time_step": 20}
```

2. Read the environment output:

```text
GET /api/text/state
```

3. Generate one command JSON object.

4. Predict without mutation:

```text
POST /api/text/validate
```

5. When valid, execute:

```text
POST /api/text/execute
```

6. Read the returned next state and repeat until `game.ended` is `true`.

This state-command-state loop is the intended foundation for connecting optimization, rule-based, reinforcement-learning, or language-model policies to DSR Simulator V23.
