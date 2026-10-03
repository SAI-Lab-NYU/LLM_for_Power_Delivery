"""Replay a recorded episode against the simulator and capture state frames.

WHY REPLAY RATHER THAN RECORD LIVE
----------------------------------
trace_<tag>.json already stores (a) the exact scenario (fault list + unknown
rectangles) and (b) the exact command batch executed at every step. So we can
reset a fresh simulator to that scenario, re-execute the recorded batches in
order, and capture a full state snapshot after each one -- no GPU, no LLM, and
identical code path for every arm (scripted / lara / ma_*).

THE BUG THIS VERSION FIXES
--------------------------
The previous version ASSUMED the scenario had been applied and never checked.
Observed symptom: two replays of completely different scenarios

    seed 1001 (2 faults, work 24) -> T5 w=5290, T6 w=5290, T10 w=5290
    seed 1010 (4 faults, work 40) -> T5 w=5290, T6 w=5290, T10 w=5290

produced byte-identical weighted-load sequences, i.e. both were silently
running whatever fault set happened to be on disk. The recorded trace for
scripted_switch_ep10 is healthy (105,190; locked=False; MPS supplies once at
T18) while its replay locked at T13 with score 0 -- a pure replay artifact that
would have been published as a visualisation of a policy failure.

Two defences, both unconditional:

  1. PRE-FLIGHT VERIFICATION. After reset we compare the live simulator state
     against the scenario: number of unknown network pixels, the set of
     currently-KNOWN faults (position + remaining work), and the unknown
     rectangles echoed back by /api/text/state. Any mismatch aborts.

  2. STEP-BY-STEP DIVERGENCE CHECK. The trace records score_after /
     weighted_after / locked_after for every step. We compare after each
     executed batch and stop at the first divergence, reporting the step. This
     is scenario-agnostic and catches any cause, known or not.

Frames are written only when the replay is faithful, unless --allow-divergence
is passed. A divergent frames file is tagged "faithful": false so the
visualizer can refuse to mislead.

OUTPUT
  frames_<tag>.json
    geometry : static node positions, line polylines, switch markers (once)
    frames   : one snapshot per step, including the batch that produced it

CAVEAT: crew/MPS/PAC start positions are NOT recorded in the trace. A run that
used --jitter-crews wrote them into RepairCrews_v1.json, so its replay cannot
be faithful; the divergence check will catch that and say so.
"""
from __future__ import annotations

import argparse
import json
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_agents.patrol import unknown_network_pixels
from llm_agents.run_experiment import free_port, start_simulator
from llm_agents.sim_api import SimulatorAPI
from llm_agents.topology import Topology

FAULT_FILE = "Faults_v1.json"
BACKUP_SUFFIX = ".replay_backup"

# Tolerances for the step-by-step divergence check.
SCORE_TOL = 1e-6
WEIGHT_TOL = 1e-6


# --------------------------------------------------------------------- #
# static geometry
# --------------------------------------------------------------------- #
def build_geometry(topo: Topology) -> Dict[str, Any]:
    """Static drawing data: bounds, nodes, line polylines, switch markers."""
    nodes = []
    for name, pos in topo.node_pos.items():
        ld = (topo.node_load or {}).get(name)
        nodes.append({
            "name": name, "x": int(pos[0]), "y": int(pos[1]),
            "p": float(ld[0]) if ld else 0.0,
            "q": float(ld[1]) if ld else 0.0,
            "w": float(ld[2]) if ld else 0.0,
            "transmission": name in topo.protected,
        })

    lines = []
    for name, px in topo.line_pixels.items():
        a, b = topo.line_ends[name]
        lines.append({"name": name, "a": a, "b": b,
                      "pts": [[int(p[0]), int(p[1])] for p in px]})

    switches = []
    for sw, lname in (topo.switch_line or {}).items():
        node = (topo.switch_node or {}).get(sw)
        if node not in topo.node_pos or lname not in topo.line_pixels:
            continue
        npos = topo.node_pos[node]
        px = topo.line_pixels[lname]
        adj = (px[1] if tuple(px[0]) == tuple(npos) and len(px) > 1
               else px[max(len(px) - 2, 0)])
        dx, dy = adj[0] - npos[0], adj[1] - npos[1]
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        switches.append({"name": sw, "line": lname, "node": node,
                         "x": npos[0] + 0.65 * dx / norm,
                         "y": npos[1] + 0.65 * dy / norm})

    return {
        "bounds": {"x_min": topo.x_min, "x_max": topo.x_max,
                   "y_min": topo.y_min, "y_max": topo.y_max},
        "nodes": nodes, "lines": lines, "switches": switches,
        "total_weighted_load": sum(abs(n["p"]) * n["w"] for n in nodes),
    }


# --------------------------------------------------------------------- #
# snapshots
# --------------------------------------------------------------------- #
def element_knowledge(state: Dict[str, Any]
                      ) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Per-node and per-line {knowledge, energized, state} from the element table."""
    rows = (((state.get("status_tables") or {}).get("elements") or {})
            .get("rows")) or []
    nodes: Dict[str, dict] = {}
    lines: Dict[str, dict] = {}
    for r in rows:
        if len(r) < 5:
            continue
        typ, name, know, power, st = r[0], r[1], str(r[2]), str(r[3]), str(r[4])
        rec = {"knowledge": know.strip().lower(),
               "energized": power.strip().lower() == "energized",
               "state": st.strip().lower()}
        if typ == "Node":
            nodes[name] = rec
        elif typ == "Line":
            lines[name] = rec
    return nodes, lines


def snapshot(state: Dict[str, Any], step: Optional[dict] = None) -> Dict[str, Any]:
    """Compact, drawable snapshot of the live simulator state."""
    game = state.get("game") or {}
    summ = state.get("summary") or {}
    nk, lk = element_knowledge(state)

    units = []
    for name, u in (state.get("objects") or {}).items():
        ctx = u.get("supply_context") or {}
        pos = u.get("position") or [0, 0]
        units.append({
            "name": name, "type": u.get("type"),
            "x": int(pos[0]), "y": int(pos[1]),
            "state": str(u.get("state") or ""),
            "range": int(u.get("inspection_range") or 0),
            "locked": bool(u.get("analysis_locked")),
            "path_left": int(u.get("remaining_path_pixels") or 0),
            "resources": u.get("remaining_resources"),
            "energy": u.get("energy"),
            "current_p": u.get("current_p"),
            "current_q": u.get("current_q"),
            "supply_node": ctx.get("node"),
            "cluster_p": ctx.get("load_p"),
            "cluster_q": ctx.get("load_q"),
            "grid_connected": bool(ctx.get("grid_connected")),
        })

    faults = []
    for f in (state.get("faults") or []):
        pos = f.get("position") or [0, 0]
        faults.append({"name": f.get("name"),
                       "x": int(pos[0]), "y": int(pos[1]),
                       "active": bool(f.get("active")),
                       "known": bool(f.get("known")),
                       "analyzed": bool(f.get("analyzed")),
                       "remaining": f.get("remaining_repair_resource")})

    switches = {nm: {"state": str(i.get("state")),
                     "commandable": bool(i.get("commandable"))}
                for nm, i in (state.get("switches") or {}).items()}

    frame = {
        "t": game.get("time_step"),
        "score": game.get("score"),
        "weighted": game.get("weighted_restored_load"),
        "total_weighted": game.get("total_weighted_load"),
        "locked": bool(game.get("locked")),
        "ended": bool(game.get("ended")),
        "energized": list(game.get("energized_loads") or []),
        "violation_loads": list(game.get("violation_loads") or []),
        "source_status": summ.get("source_status"),
        "source_restore_step": summ.get("source_restore_step"),
        "units": units, "faults": faults, "switches": switches,
        "node_state": nk, "line_state": lk,
    }
    if step is not None:
        frame.update({
            "command": step.get("command"),
            "source": step.get("source"),
            "message": step.get("message"),
            "executed_ok": step.get("executed_ok"),
            "delta": step.get("delta"),
            "reasoning": step.get("reasoning"),
            "n_plans": step.get("n_plans"),
            "replanned": step.get("replanned"),
            "switch_report": step.get("switch_report"),
            "terminal_fatal": step.get("terminal_fatal"),
        })
    return frame


# --------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------- #
def _f(x, default: float = 0.0) -> float:
    try:
        return default if x is None else float(x)
    except (TypeError, ValueError):
        return default


def verify_scenario(state: Dict[str, Any], scen: Dict[str, Any],
                    topo: Topology) -> List[str]:
    """Does the freshly reset simulator actually hold the recorded scenario?

    The text state exposes only KNOWN faults, so we cannot count them all.
    What we CAN check, and all of which would have caught the observed bug:
      * every known fault's position must appear in the scenario's fault list;
      * a known fault's repair_resource (once analysed) must match;
      * the unknown rectangles echoed back must match the scenario;
      * the count of unknown network pixels must be > 0 if rectangles exist.
    """
    problems: List[str] = []

    want = {(int(f["position"][0]), int(f["position"][1])):
            int(f.get("repair_resource") or 0)
            for f in (scen.get("faults") or [])}
    if not want:
        problems.append("scenario has NO faults recorded; cannot verify")

    for f in (state.get("faults") or []):
        pos = f.get("position") or [0, 0]
        key = (int(pos[0]), int(pos[1]))
        if key not in want:
            problems.append(
                f"simulator reports a fault at {list(key)} that is NOT in the "
                f"recorded scenario {sorted(want)} -- the wrong Faults file is "
                f"loaded")
            continue
        rec = f.get("repair_resource")
        if rec is not None and int(rec) != want[key]:
            problems.append(f"fault at {list(key)}: simulator says "
                            f"repair_resource={rec}, scenario says {want[key]}")

    got_rects = state.get("unknown_regions") or []
    want_rects = scen.get("unknown_regions") or []

    def norm(r):
        return (list(map(int, r.get("lower_left", [0, 0]))),
                list(map(int, r.get("upper_right", [0, 0]))))

    if sorted(map(norm, got_rects)) != sorted(map(norm, want_rects)):
        problems.append(f"unknown rectangles differ: simulator "
                        f"{[norm(r) for r in got_rects]} vs scenario "
                        f"{[norm(r) for r in want_rects]}")

    if want_rects:
        n_unknown = len(unknown_network_pixels(state, topo))
        if n_unknown == 0:
            problems.append("scenario defines unknown rectangles but the "
                            "simulator reports 0 unknown network pixels")
    return problems


def compare_step(frame: Dict[str, Any], row: Dict[str, Any]) -> Optional[str]:
    """Did the replayed step reproduce the recorded outcome?"""
    rec_score = row.get("score_after")
    rec_w = row.get("weighted_after")
    rec_locked = row.get("locked_after")

    if rec_score is not None and abs(_f(frame["score"]) - _f(rec_score)) > SCORE_TOL:
        return (f"score {frame['score']} != recorded {rec_score}")
    if rec_w is not None and abs(_f(frame["weighted"]) - _f(rec_w)) > WEIGHT_TOL:
        return (f"weighted {frame['weighted']} != recorded {rec_w}")
    if rec_locked is not None and bool(frame["locked"]) != bool(rec_locked):
        return (f"locked {frame['locked']} != recorded {rec_locked}")
    return None


# --------------------------------------------------------------------- #
# replay
# --------------------------------------------------------------------- #
def replay(trace_path: Path, sim_dir: Path, out_path: Optional[Path] = None,
           allow_divergence: bool = False, verbose: bool = True
           ) -> Optional[Path]:
    trace = json.loads(Path(trace_path).read_text(encoding="utf-8"))
    summary = trace.get("summary") or {}
    scen = trace.get("scenario") or {}
    rows = trace.get("trace") or []
    if not rows:
        raise RuntimeError(f"{trace_path} has no trace rows")
    if not scen.get("faults"):
        raise RuntimeError(f"{trace_path} has no scenario fault list; this "
                           f"trace predates scenario recording and cannot be "
                           f"replayed faithfully")

    topo = Topology(sim_dir)
    geometry = build_geometry(topo)

    # ---- install the scenario's faults, with an always-restored backup ----
    fault_path = sim_dir / FAULT_FILE
    backup_path = sim_dir / (FAULT_FILE + BACKUP_SUFFIX)
    if backup_path.exists():
        # A previous replay died before restoring. Put it back FIRST so we
        # never carry a stale file forward, then proceed normally.
        print(f"[replay] found stale {backup_path.name}; restoring it first",
              flush=True)
        shutil.copy2(backup_path, fault_path)
        backup_path.unlink()
    shutil.copy2(fault_path, backup_path)
    fault_path.write_text(json.dumps({"Faults": scen["faults"]}, indent=2))
    if verbose:
        print(f"[replay] installed {len(scen['faults'])} faults: "
              f"{[f['name'] + str(f['position']) for f in scen['faults']]}",
              flush=True)

    port = free_port()
    sim = start_simulator(sim_dir, port)
    api = SimulatorAPI(f"http://127.0.0.1:{port}")
    frames: List[Dict[str, Any]] = []
    problems: List[str] = []
    diverged_at: Optional[int] = None

    try:
        api.reset(max_time_step=int(summary.get("max_time_step") or 20),
                  unknown_regions=scen.get("unknown_regions"))
        state0 = api.state()

        # ---- PRE-FLIGHT: is the scenario actually loaded? -------------
        problems = verify_scenario(state0, scen, topo)
        if problems:
            print("[replay] *** SCENARIO VERIFICATION FAILED ***", flush=True)
            for p in problems:
                print(f"[replay]   - {p}", flush=True)
            if not allow_divergence:
                raise RuntimeError(
                    "the simulator is not holding the recorded scenario; "
                    "refusing to produce a misleading visualisation. Check "
                    f"{fault_path} and any *.orig_backup / *.replay_backup "
                    "left behind by an interrupted run.")

        frames.append(snapshot(state0))          # T0, before any command
        if verbose:
            g0 = state0.get("game") or {}
            print(f"[replay] T0 captured (score={g0.get('score')} "
                  f"weighted={g0.get('weighted_restored_load')})", flush=True)

        # ---- re-execute the recorded batches --------------------------
        for row in rows:
            batch = row.get("command") or {"switch_commands": {},
                                           "mobile_commands": {}}
            res = api.execute(batch)
            if not res.get("ok"):
                # the original runner fell back to a no-op here; mirror it
                res = api.execute({"switch_commands": {},
                                   "mobile_commands": {}})
            st = api.state()
            frame = snapshot(st, row)
            frames.append(frame)

            why = compare_step(frame, row)
            if verbose:
                flag = "" if why is None else f"   <== DIVERGED: {why}"
                print(f"[replay] T{row.get('t')} -> score={frame['score']} "
                      f"weighted={frame['weighted']} "
                      f"locked={frame['locked']}{flag}", flush=True)
            if why is not None and diverged_at is None:
                diverged_at = int(row.get("t", -1))
                problems.append(f"step T{diverged_at}: {why}")
                if not allow_divergence:
                    break

            g = st.get("game") or {}
            if g.get("ended") or not g.get("commands_enabled"):
                break
    finally:
        if backup_path.exists():
            shutil.copy2(backup_path, fault_path)
            backup_path.unlink()
        sim.send_signal(signal.SIGTERM)
        try:
            sim.wait(timeout=15)
        except subprocess.TimeoutExpired:
            sim.kill()

    faithful = diverged_at is None and not problems
    tag = Path(trace_path).stem
    tag = tag[len("trace_"):] if tag.startswith("trace_") else tag

    if not faithful:
        print(f"[replay] NOT FAITHFUL -- "
              f"{'diverged at T' + str(diverged_at) if diverged_at is not None else 'verification failed'}",
              flush=True)
        for p in problems:
            print(f"[replay]   - {p}", flush=True)
        if not allow_divergence:
            print(f"[replay] frames NOT written. Re-run with "
                  f"--allow-divergence to inspect the divergent trajectory.",
                  flush=True)
            return None

    out_path = out_path or Path(trace_path).parent / f"frames_{tag}.json"
    out_path.write_text(json.dumps({
        "meta": {"tag": tag, "method": summary.get("method"),
                 "episode": summary.get("episode"),
                 "trial": summary.get("trial"),
                 "model": summary.get("model"),
                 "final_score": summary.get("final_score"),
                 "steps_taken": summary.get("steps_taken"),
                 "locked": summary.get("locked"),
                 "end_reason": summary.get("end_reason"),
                 "max_time_step": summary.get("max_time_step"),
                 "faithful": faithful,
                 "diverged_at": diverged_at,
                 "problems": problems},
        "scenario": scen, "geometry": geometry, "frames": frames,
    }, indent=1, default=str))

    final = frames[-1].get("score")
    rec = summary.get("final_score")
    status = "MATCH" if faithful else f"MISMATCH (recorded {rec})"
    print(f"[replay] wrote {out_path}  frames={len(frames)}  "
          f"final_score={final}  [{status}]", flush=True)
    return out_path


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="trace_*.json, or a results directory")
    ap.add_argument("--sim-dir", default="dsr_sim_v23")
    ap.add_argument("--method", default=None,
                    help="when PATH is a directory, filter by method")
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="replay at most N traces (0 = all matched)")
    ap.add_argument("--allow-divergence", action="store_true",
                    help="write frames even when the replay is not faithful")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        pat = f"trace_{args.method}_ep*.json" if args.method else "trace_*.json"
        files = sorted(p.glob(pat))
        if args.episode is not None:
            files = [f for f in files if f"_ep{args.episode}_" in f.name]
    else:
        files = [p]
    if args.limit:
        files = files[:args.limit]
    if not files:
        print(f"No traces matched under {p}")
        return

    sim_dir = Path(args.sim_dir).resolve()
    ok = bad = 0
    for f in files:
        print(f"\n=== replaying {f.name} ===")
        try:
            res = replay(f, sim_dir, allow_divergence=args.allow_divergence,
                         verbose=not args.quiet)
            ok += int(res is not None)
            bad += int(res is None)
        except Exception as error:
            bad += 1
            print(f"[error] {f.name}: {error}")
    print(f"\n{ok} faithful, {bad} failed/divergent")


if __name__ == "__main__":
    main()