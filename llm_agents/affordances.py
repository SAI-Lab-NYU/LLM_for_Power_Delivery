"""Legal-action menus, prescriptive recommendations, and agent-facing views.

Three responsibilities:
  1. build_affordances(state)  -- vetted option tables from the simulator's own
                                  bookkeeping; no ID can be hallucinated.
  2. recommend() / recommended_command()  -- deterministic greedy policy; this
                                  IS the `scripted` baseline and also the
                                  "strong prior" shown to every LLM agent.
  3. team_digest() / unit_block() / situation_card()  -- the three text views.

Keeping hard feasibility in the environment rather than in the LLM is the
design LARA reports as yielding zero constraint violations [1]; this module is
where that separation lives.

THE MPS FAILURE MODES
---------------------
Continuity makes any transient island feed catastrophic: once a load is
energized it must stay energized at every later completed step, or the score
locks to zero permanently.

  (a) STARTING a supply the MPS cannot hold to the horizon.
      MPS1 has energy=8000; the 800 kW island costs 800/step = exactly 10 steps
      on a 20-step horizon. Supplying from T0 guarantees auto-disconnect at
      ~T10 -> island dark -> score 0.
      GATE: energy >= p * (steps_remaining + SUPPLY_MARGIN_STEPS).

  (b) CONTINUING with exhausted energy. Once Supplying, the only legal command
      is `continue`; if energy < p the simulator auto-disconnects on the last
      frame regardless. Unrecoverable; DETECTED, not preventable.

  (c) THE ISLAND GROWING UNDER A SUPPLYING MPS.
      Observed on seed 1010. Fault B at (-4,16) sits on the BODY of L4
      (N2<->N6). L4 and L5 have NO switch on either terminal, so repairing B
      re-closed both unconditionally and merged N6 (10,5,4) and N7 (40,15,5)
      into MPS1's island: 250/120 -> 300/140 while MPS1 still output 250/120.
      The islanded balance rule was then violated BY THE STATE ITSELF, so the
      simulator rejected every batch including a no-op, and the clock froze.

WHY THE OLD REPAIR INTERLOCK COULD NOT WORK
-------------------------------------------
It iterated state["faults"], which lists only KNOWN faults. On seed 1010 fault
B was undiscovered at T0 (analysed only at T11), so no blocker was visible and
the 250 kW supply was allowed. You cannot interlock against a fault you have
not observed -- the Dec-POMDP partial-observability problem [5]. Worse, where
the interlock DID fire (seed 1017) it deadlocked: it refused a repair to
protect a supplying MPS that can only `continue`, so neither side ever yielded
and 14 repairs were blocked, costing the early-completion bonus.

THE FIX: FAULT-CLOSURE GATING
-----------------------------
fault_closure_island() computes, from TOPOLOGY ALONE, the nodes reachable from
the MPS node through conducting terminals (switchless, or switch closed),
IGNORING fault state. That is the worst-case island the MPS may end up serving
once repairs complete.

  * An MPS will not START supplying while its current cluster is SMALLER than
    its fault closure. A smaller cluster means something -- a fault, known or
    unknown -- is isolating part of the closure, and it will be repaired.
  * Sustainability is checked against the CLOSURE load, not the momentary one.
  * The repair interlock is now ADVISORY ONLY (reported, never enforced),
    which removes the livelock. Switch-driven growth needs no interlock: the
    SwitchOperator validates every toggle through /api/text/validate, so an
    imbalancing closure is filtered before it can be chosen.

Seed 1010 at T0: cluster {N2}=250 kW, closure {N2,N6,N7}=300 kW -> HOLD. The
undiscovered fault never has to be seen.

MODE (c) IS ONLY FATAL WHEN ISLANDED
------------------------------------
The simulator has two balance rules:
    grid_connected -> P_grid = P_load - sum(P_MPS); ANY mps_p is valid
    islanded       -> sum(P_MPS) == cluster load EXACTLY
The grid returns at step 6, after which a mismatch is harmless. Checking it
unconditionally produced false "unrecoverable" flags on 7 of 20 episodes that
went on to score 140k-167k. Mode (c) is therefore gated on NOT grid_connected.
"""
from __future__ import annotations

import json
from collections import deque
from math import hypot
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_agents.patrol import coverage_route, unknown_areas, unknown_network_pixels

# --------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------- #
# Extra steps of energy headroom before an MPS may START supplying. Must be
# >= 1: the simulator checks `energy + 1e-9 < current_p` on the LAST frame of a
# step, so landing on exactly zero disconnects on the final scored step.
SUPPLY_MARGIN_STEPS = 1

# Tolerance for islanded P/Q balance comparisons. The simulator uses 1e-6.
BALANCE_TOL = 1e-4

# Treat open-but-commandable switches as potentially closable when computing
# the fault closure. Default False: switch-driven growth is already filtered by
# the SwitchOperator's validate() gate, and including them is very
# conservative (it would often forbid supplying at all).
CLOSURE_INCLUDES_COMMANDABLE = False

# Enforce the repair interlock (remove destabilising repairs from the menu).
# Default False -- it livelocks, and the closure gate makes it unnecessary.
# Blockers are still REPORTED in affordances["blocked_repairs"].
ENFORCE_REPAIR_INTERLOCK = False

# --------------------------------------------------------------------- #
_TOPO = None


def set_topology(topo) -> None:
    global _TOPO
    _TOPO = topo


def get_topology():
    return _TOPO


NODE_POSITIONS: Dict[str, Tuple[int, int]] = {
    "N1": (0, 20),  "N2": (0, 16),   "N3": (0, 0),    "N4": (8, 16),
    "N5": (12, 12), "N6": (-8, 16),  "N7": (-10, 16), "N8": (4, 0),
    "N9": (12, 0),  "N10": (-4, 0),  "N11": (-4, -8), "N12": (-8, 0),
    "N13": (0, -16), "V1": (-10, 20), "V2": (-12, 0),
    "V3": (-12, -8), "V4": (10, -16),
}
MAP_X = (-15, 15)
MAP_Y = (-18, 22)


# --------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------- #
def _pos(raw) -> Tuple[int, int]:
    try:
        return int(raw[0]), int(raw[1])
    except Exception:
        return 0, 0


def _manhattan(a, b) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _clamp(p, topo=None) -> List[int]:
    topo = topo or _TOPO
    if topo is not None:
        return topo.clamp(p)
    return [max(MAP_X[0], min(MAP_X[1], int(p[0]))),
            max(MAP_Y[0], min(MAP_Y[1], int(p[1])))]


def _node_positions(topo=None) -> Dict[str, Tuple[int, int]]:
    topo = topo or _TOPO
    return dict(topo.node_pos) if topo is not None else dict(NODE_POSITIONS)


def _norm_state(unit: dict) -> str:
    return str(unit.get("state", "")).strip().lower()


def _is_supplying(unit: dict) -> bool:
    return _norm_state(unit) == "supplying"


def _is_moving(unit: dict) -> bool:
    # BUGFIX: the simulator reports "Moving" (capitalised).
    return _norm_state(unit) == "moving"


def _unknown_region_centres(state: dict) -> List[Tuple[int, int]]:
    out = []
    for r in state.get("unknown_regions", []) or []:
        try:
            ll, ur = r["lower_left"], r["upper_right"]
            out.append(((int(ll[0]) + int(ur[0])) // 2,
                        (int(ll[1]) + int(ur[1])) // 2))
        except Exception:
            continue
    return out


def _load_value(item: dict) -> float:
    try:
        return abs(float(item.get("p") or item.get("p_kw") or 0.0)) * \
               float(item.get("w") or item.get("weight") or 0.0)
    except Exception:
        return 0.0


def _f(x, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def horizon(state: dict) -> Tuple[int, int, int]:
    """Return (current_time_step, max_time_step, steps_remaining)."""
    game = state.get("game", {}) or {}
    try:
        t = int(game.get("time_step") or 0)
    except (TypeError, ValueError):
        t = 0
    try:
        n = int(game.get("max_time_step") or 20)
    except (TypeError, ValueError):
        n = 20
    return t, n, max(0, n - t)


# --------------------------------------------------------------------- #
# topology reasoning
# --------------------------------------------------------------------- #
def area_rows(state: dict) -> List[Dict[str, str]]:
    """Normalized rows of status_tables['areas']."""
    return unknown_areas(state)


def island_nodes_for_areas(state: dict, area_names) -> Set[str]:
    """Union of node names across the named static Areas."""
    want = {str(a).strip() for a in (area_names or []) if str(a).strip()}
    if not want:
        return set()
    out: Set[str] = set()
    for row in area_rows(state):
        if str(row.get("area", "")).strip() in want:
            out.update(n.strip() for n in str(row.get("nodes", "")).split(",")
                       if n.strip())
    return out


def _terminal_conducts(line: str, node: str, switches: Dict[str, Any], topo,
                       include_commandable: bool = False) -> bool:
    """Would the (line, node) terminal conduct, ignoring fault state?

    A terminal with NO switch is permanently connected -- exactly why repairing
    a fault on L4/L5 re-closed them and grew MPS1's island on seed 1010.
    """
    if topo is None:
        return True
    for sw_name, sw_line in (getattr(topo, "switch_line", {}) or {}).items():
        if sw_line != line:
            continue
        if (getattr(topo, "switch_node", {}) or {}).get(sw_name) != node:
            continue
        info = (switches or {}).get(sw_name) or {}
        st = str(info.get("state", "")).strip().lower()
        if st == "closed":
            return True
        if include_commandable and bool(info.get("commandable")):
            return True        # could be closed later by the SwitchOperator
        return False           # "opened" / "unknown"
    return True                # no switch on this terminal -> always conducts


def fault_closure_island(start_node: Optional[str], switches: Dict[str, Any],
                         topo, include_commandable: bool = CLOSURE_INCLUDES_COMMANDABLE
                         ) -> Set[str]:
    """Nodes reachable from `start_node` IGNORING FAULTS.

    This is the worst-case island an MPS at `start_node` may end up serving
    once every fault on the way is repaired. Computed from topology alone, so
    it is correct even for faults the team has not discovered yet.
    """
    if topo is None or not start_node:
        return set()
    ends = getattr(topo, "line_ends", {}) or {}
    if start_node not in (getattr(topo, "node_pos", {}) or {}):
        return set()

    adj: Dict[str, List[str]] = {}
    for lname, (a, b) in ends.items():
        if not _terminal_conducts(lname, a, switches, topo, include_commandable):
            continue
        if not _terminal_conducts(lname, b, switches, topo, include_commandable):
            continue
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)

    seen = {start_node}
    q = deque([start_node])
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)
    return seen


def island_load(nodes, topo) -> Tuple[float, float]:
    """Total (P, Q) of the load-bearing nodes in `nodes`."""
    if topo is None:
        return 0.0, 0.0
    loads = getattr(topo, "node_load", {}) or {}
    p = q = 0.0
    for n in (nodes or []):
        ld = loads.get(n)
        if not ld:
            continue
        p += _f(ld[0])
        q += _f(ld[1])
    return p, q


def repair_merges_into_island(fault_position, island: Set[str],
                              switches: Dict[str, Any],
                              topo) -> Tuple[bool, Optional[str], Optional[str]]:
    """Would clearing a fault at this pixel merge NEW load into `island`?

    Advisory diagnostic only -- it can only see KNOWN faults, which is why it
    is no longer the primary safeguard. See fault_closure_island().
    """
    if topo is None or not island:
        return False, None, None
    try:
        fp = (int(fault_position[0]), int(fault_position[1]))
    except Exception:
        return False, None, None

    for lname, body in (getattr(topo, "line_body", {}) or {}).items():
        if fp not in set(body):
            continue
        ends = (getattr(topo, "line_ends", {}) or {}).get(lname)
        if not ends:
            continue
        a, b = ends
        if not _terminal_conducts(lname, a, switches, topo):
            continue
        if not _terminal_conducts(lname, b, switches, topo):
            continue
        in_a, in_b = (a in island), (b in island)
        if in_a != in_b:
            return True, lname, (b if in_a else a)
    return False, None, None


# --------------------------------------------------------------------- #
# MPS supply planning
# --------------------------------------------------------------------- #
def mps_supply_plan(state: dict, unit: dict, name: str = "MPS",
                    topo=None) -> Dict[str, Any]:
    """Decide whether this MPS may legally AND safely supply."""
    topo = topo or _TOPO
    t, n, remaining = horizon(state)
    switches = state.get("switches", {}) or {}
    faults = state.get("faults", []) or []

    ctx = unit.get("supply_context") or {}
    p = _f(ctx.get("load_p"))            # CURRENT cluster demand
    q = _f(ctx.get("load_q"))
    p_lim = _f(unit.get("p_limit"))
    s_lim = _f(unit.get("s_limit"))
    energy = _f(unit.get("energy"))
    supplying = _is_supplying(unit)
    grid_connected = bool(ctx.get("grid_connected"))
    cur_p = _f(unit.get("current_p"))
    cur_q = _f(unit.get("current_q"))
    node = ctx.get("node")
    areas = list(ctx.get("areas") or [])
    island = island_nodes_for_areas(state, areas)

    closure = fault_closure_island(node, switches, topo)
    clo_p, clo_q = island_load(closure, topo)

    out: Dict[str, Any] = {
        "unit": name,
        "supplying": supplying,
        "node": node,
        "areas": areas,
        "island_nodes": sorted(island),
        "grid_connected": grid_connected,
        # mode (a)
        "feasible_now": False, "sustainable": False,
        "p": p, "q": q, "energy": energy, "energy_needed": 0.0,
        "start_step": None,
        # mode (b)
        "current_p": cur_p, "current_q": cur_q,
        "can_sustain_one_step": True,
        "steps_of_energy_left": None,
        "will_die_at_step": None,
        # mode (c)
        "island_changed": False,
        "required_p": p, "required_q": q,
        # closure gate
        "closure_nodes": sorted(closure),
        "closure_p": clo_p, "closure_q": clo_q,
        "island_unstable": False,
        # advisory interlock
        "blocked_by_repair": False,
        "blocking_faults": [],
        "reason": "",
    }

    # ---------------- already supplying -----------------------------
    if supplying:
        # mode (c) -- ONLY fatal when islanded. When the grid is connected it
        # balances the difference, so a mismatch is harmless. Checking this
        # unconditionally produced false "unrecoverable" flags on 7 of 20
        # episodes that went on to score 140k-167k.
        if (not grid_connected) and cur_p > 0.0 and (
                abs(cur_p - p) > BALANCE_TOL or abs(cur_q - q) > BALANCE_TOL):
            out["island_changed"] = True
            out["reason"] = (
                f"TERMINAL: the ISLANDED cluster under {name} CHANGED. It "
                f"outputs P={cur_p:g} Q={cur_q:g} but its cluster now demands "
                f"P={p:g} Q={q:g}. A repair re-closed a switchless line and "
                f"merged new load in. A Supplying MPS can only `continue`, so "
                f"the imbalance cannot be corrected: the simulator will reject "
                f"EVERY command, including a no-op.")
            return out

        # mode (b) -- energy
        if cur_p <= 0.0:
            out["reason"] = f"{name} supplying with zero P output (no drain)"
            return out
        steps_left = int(energy // cur_p)
        out["steps_of_energy_left"] = steps_left
        out["can_sustain_one_step"] = (energy + 1e-9 >= cur_p)
        out["will_die_at_step"] = t + steps_left
        if not out["can_sustain_one_step"]:
            out["reason"] = (
                f"TERMINAL: {name} supplying P={cur_p:g} with only {energy:g} "
                f"energy left. The simulator auto-disconnects on the last "
                f"frame of THIS step; its island goes dark and the score "
                f"locks. No command can prevent this.")
        elif steps_left < remaining and not grid_connected:
            out["reason"] = (
                f"{name} supplying P={cur_p:g}; energy {energy:g} lasts "
                f"{steps_left} more steps but {remaining} remain. The island "
                f"WILL go dark at ~step {out['will_die_at_step']}.")
        else:
            out["reason"] = (f"{name} supplying P={cur_p:g} sustainably "
                             f"(grid_connected={grid_connected}, "
                             f"{steps_left} steps of energy, {remaining} left)")
        return out

    # ---------------- not yet supplying: may it START? ---------------
    if not ctx.get("available") or not node:
        out["reason"] = (f"{name} is not at a safe Node with an islanded "
                         f"cluster ({ctx.get('message') or 'no context'})")
        return out
    if p <= 0.0:
        out["reason"] = "connected cluster has zero active load: nothing to feed"
        return out
    if p > p_lim + 1e-9:
        out["reason"] = f"cluster P={p:g} exceeds P_limit={p_lim:g}"
        return out
    if hypot(p, q) > s_lim + 1e-9:
        out["reason"] = f"cluster S={hypot(p, q):g} exceeds S_limit={s_lim:g}"
        return out

    out["feasible_now"] = True

    # ---- advisory interlock (reported; enforced only if configured) --
    blockers: List[Dict[str, Any]] = []
    for fl in faults:
        if not fl.get("active"):
            continue
        merges, lname, joined = repair_merges_into_island(
            fl.get("position"), island, switches, topo)
        if merges:
            blockers.append({"fault": fl.get("name"), "line": lname,
                             "joins": joined})
    if blockers:
        out["blocking_faults"] = blockers
        out["blocked_by_repair"] = bool(ENFORCE_REPAIR_INTERLOCK)

    # ---- THE CLOSURE GATE -------------------------------------------
    # If the current cluster is SMALLER than its fault closure, something is
    # isolating part of the closure -- a fault, possibly undiscovered. That
    # fault will be repaired, the switchless line will re-close, and the load
    # will merge under a supplying MPS that cannot be re-dispatched. Wait.
    if closure and clo_p > p + BALANCE_TOL:
        out["island_unstable"] = True
        extra = sorted(set(out["closure_nodes"]) - island)
        out["reason"] = (
            f"HOLD {name}: the island at {node} is only P={p:g} right now, but "
            f"its FAULT CLOSURE is P={clo_p:g} Q={clo_q:g} over "
            f"{out['closure_nodes']}. Nodes {extra} are reachable through "
            f"switchless lines and are currently isolated only by a fault "
            f"(possibly one we have not discovered yet). Repairing it will "
            f"merge that load and break the islanded balance under a supplying "
            f"MPS, locking the score to zero. Repair first, then supply the "
            f"stable P={clo_p:g} island.")
        return out

    # ---- mode (a): horizon sustainability, against the CLOSURE load --
    gate_p = max(p, clo_p)
    steps_to_cover = remaining + SUPPLY_MARGIN_STEPS
    need = gate_p * steps_to_cover
    out["energy_needed"] = need

    if energy + 1e-9 >= need:
        if not out["blocked_by_repair"]:
            out["sustainable"] = True
        out["reason"] = (
            f"island at {node} needs exact P={p:g} Q={q:g} (closure "
            f"P={clo_p:g}); energy {energy:g} >= {need:g} covers all "
            f"{remaining} remaining steps (+{SUPPLY_MARGIN_STEPS} margin)")
        if out["blocked_by_repair"]:
            out["reason"] = ("INTERLOCK HOLD (enforced): " + out["reason"]
                             + "; a pending repair would grow this island")
        return out

    t_start = n + SUPPLY_MARGIN_STEPS - int(energy // gate_p)
    out["start_step"] = max(t, t_start)
    out["reason"] = (
        f"UNSUSTAINABLE NOW: island needs P={gate_p:g}/step for {remaining} "
        f"remaining steps = {need:g}, but only {energy:g} stored. Supplying "
        f"now would auto-disconnect around step {t + int(energy // gate_p)}, "
        f"de-energize the island, and PERMANENTLY LOCK the score to zero. "
        f"Wait until step ~{out['start_step']}, then supply to the end.")
    return out


def terminal_fatal_report(state: dict,
                          affordances: Optional[dict] = None,
                          topo=None) -> Dict[str, Any]:
    """States where NO command can avoid a continuity violation.

    When this fires the episode is already lost; the correct response is to
    record it and stop, never to retry.
    """
    plans = (affordances or {}).get("supply_plans")
    if plans is None:
        plans = {nm: mps_supply_plan(state, u, nm, topo)
                 for nm, u in (state.get("objects") or {}).items()
                 if u.get("type") == "MPS"}

    doomed, warn = [], []
    _, _, remaining = horizon(state)
    for nm, plan in plans.items():
        if not plan.get("supplying"):
            continue
        if plan.get("island_changed"):
            doomed.append({"unit": nm, "mode": "island_changed",
                           "reason": plan.get("reason", ""),
                           "current_p": plan.get("current_p"),
                           "required_p": plan.get("required_p")})
        elif not plan.get("can_sustain_one_step", True):
            doomed.append({"unit": nm, "mode": "energy_exhausted",
                           "reason": plan.get("reason", ""),
                           "current_p": plan.get("current_p"),
                           "energy": plan.get("energy")})
        elif (plan.get("steps_of_energy_left") is not None
              and plan["steps_of_energy_left"] < remaining
              and not plan.get("grid_connected")):
            warn.append({"unit": nm, "dies_at": plan.get("will_die_at_step"),
                         "reason": plan.get("reason", "")})

    game = state.get("game", {}) or {}
    return {
        "terminal_fatal": bool(doomed),
        "already_locked": bool(game.get("locked")),
        "doomed_units": doomed,
        "doomed_soon": warn,
        "summary": ("; ".join(d["reason"] for d in doomed) if doomed else
                    "; ".join(w["reason"] for w in warn) if warn else ""),
    }


# --------------------------------------------------------------------- #
# 1. affordance tables
# --------------------------------------------------------------------- #
def build_affordances(state: dict) -> Dict[str, Any]:
    topo = _TOPO
    objects = state.get("objects", {}) or {}
    faults = state.get("faults", []) or []
    switches = state.get("switches", {}) or {}
    loads = state.get("loads", []) or []

    known_active = [f for f in faults if f.get("active") and f.get("known")]
    unknown_px = unknown_network_pixels(state, topo) if topo is not None else []
    areas = area_rows(state)

    supply_plans: Dict[str, Dict[str, Any]] = {}
    for nm, u in objects.items():
        if u.get("type") == "MPS":
            supply_plans[nm] = mps_supply_plan(state, u, nm, topo)

    # Islands fed by a Supplying MPS, for the advisory interlock report.
    protected_islands: List[Tuple[str, Set[str]]] = [
        (nm, set(pl.get("island_nodes") or []))
        for nm, pl in supply_plans.items() if pl.get("supplying")
    ]

    options: Dict[str, List[dict]] = {}
    move_targets: Dict[str, List[dict]] = {}
    blocked_repairs: List[Dict[str, Any]] = []

    fault_pos = {str(f.get("name")): f.get("position") for f in faults}

    for name, unit in objects.items():
        if unit.get("analysis_locked"):
            options[name] = [{"type": "continue"}]
            move_targets[name] = []
            continue
        if unit.get("type") == "MPS" and _is_supplying(unit):
            options[name] = [{"type": "continue"}]
            move_targets[name] = []
            continue

        menu: List[dict] = [{"type": "continue"}, {"type": "stay"}]

        # ---- repair -------------------------------------------------
        for rep in unit.get("repairable_faults", []) or []:
            amount = int(rep.get("maximum_amount") or 0)
            if amount <= 0:
                continue
            fname = str(rep.get("name"))
            flagged = None
            for mps_name, island in protected_islands:
                merges, lname, joined = repair_merges_into_island(
                    fault_pos.get(fname), island, switches, topo)
                if merges:
                    flagged = {"fault": fname, "crew": name, "mps": mps_name,
                               "line": lname, "joins": joined,
                               "enforced": bool(ENFORCE_REPAIR_INTERLOCK)}
                    break
            if flagged:
                blocked_repairs.append(flagged)
                if ENFORCE_REPAIR_INTERLOCK:
                    continue        # removed from the menu
            menu.append({"type": "repair", "fault": fname, "amount": amount})

        # ---- MPS supply ---------------------------------------------
        if unit.get("type") == "MPS":
            plan = supply_plans.get(name) or {}
            if (plan.get("feasible_now") and plan.get("sustainable")
                    and not plan.get("island_unstable")
                    and not plan.get("blocked_by_repair")):
                menu.append({"type": "supply",
                             "p": plan["p"], "q": plan["q"]})

        options[name] = menu

        # ---- move shortlist -----------------------------------------
        here = _pos(unit.get("position"))
        targets: List[dict] = []
        for f in known_active:
            fp = _pos(f.get("position"))
            targets.append({
                "waypoint": _clamp(fp, topo),
                "dist": _manhattan(here, fp),
                "why": f"fault {f['name']}"
                       f"{'' if f.get('analyzed') else ' (arrive -> 1 locked analysis step)'}"})

        radius = int(unit.get("inspection_range") or 0)
        if topo is not None and unknown_px:
            route, covered = coverage_route(here, unknown_px, radius, topo,
                                            max_waypoints=1)
            if route:
                targets.append({
                    "waypoint": route[0],
                    "dist": _manhattan(here, route[0]),
                    "why": f"patrol: reveals ~{covered} unknown network pixels"})
        else:
            for c in _unknown_region_centres(state):
                targets.append({"waypoint": _clamp(c, topo),
                                "dist": _manhattan(here, c),
                                "why": "patrol unknown region"})

        targets.sort(key=lambda t: t["dist"])
        move_targets[name] = targets[:3]

    switch_options = {
        nm: ("closed" if info.get("state") == "opened" else "opened")
        for nm, info in switches.items() if info.get("commandable")
    }
    unserved = [{"node": it.get("node"), "p": it.get("p_kw"),
                 "w": it.get("weight")}
                for it in loads if not it.get("energized")]
    t, n, remaining = horizon(state)

    aff = {
        "object_options": options,
        "move_targets": move_targets,
        "switch_toggle_options": switch_options,
        "unenergized_loads": unserved,
        "known_active_faults": known_active,
        "unknown_network_pixels": unknown_px,
        "areas": areas,
        "ready_areas": [a for a in areas
                        if str(a.get("energizable", "")).strip().lower() == "yes"],
        "supply_plans": supply_plans,
        "blocked_repairs": blocked_repairs,
        "protected_islands": {nm: sorted(isl) for nm, isl in protected_islands},
        "time_step": t, "max_time_step": n, "steps_remaining": remaining,
    }
    aff["terminal"] = terminal_fatal_report(state, aff, topo)
    return aff


# --------------------------------------------------------------------- #
# 2. recommender (also the scripted baseline)
# --------------------------------------------------------------------- #
def recommend(state: dict, affordances: dict,
              topo=None) -> Dict[str, Tuple[dict, str]]:
    topo = topo or _TOPO
    objects = state.get("objects", {}) or {}
    faults = affordances["known_active_faults"]
    unserved = affordances["unenergized_loads"]
    supply_plans = affordances.get("supply_plans") or {}
    node_pos = _node_positions(topo)

    recs: Dict[str, Tuple[dict, str]] = {}
    claimed: set = set()
    unknown_px = list(affordances.get("unknown_network_pixels") or [])
    centres = _unknown_region_centres(state)

    for name in sorted(objects):
        unit = objects[name]
        utype = unit.get("type")
        here = _pos(unit.get("position"))
        radius = int(unit.get("inspection_range") or 0)

        # 1) mandatory analysis step
        if unit.get("analysis_locked"):
            recs[name] = ({"type": "continue"},
                          "analysis-locked: ONLY continue is legal this step")
            continue

        # 2) a Supplying MPS can only continue
        if utype == "MPS" and _is_supplying(unit):
            plan = supply_plans.get(name) or {}
            if plan.get("island_changed") or not plan.get("can_sustain_one_step", True):
                recs[name] = ({"type": "continue"},
                              f"TERMINAL (unavoidable): {plan.get('reason','')}")
            else:
                recs[name] = ({"type": "continue"},
                              "already supplying: moving/staying disconnects "
                              "it, de-energizes its island, and locks the score")
            continue

        # 3) already en route -> let it finish
        if _is_moving(unit) and int(unit.get("remaining_path_pixels") or 0) > 0:
            recs[name] = ({"type": "continue"},
                          f"en route, {unit['remaining_path_pixels']} px remaining")
            continue

        # 4) RC standing on a legally repairable fault
        if utype == "RC":
            reps = [r for r in (unit.get("repairable_faults") or [])
                    if int(r.get("maximum_amount") or 0) > 0]
            legal = {o.get("fault") for o in
                     affordances["object_options"].get(name, [])
                     if o.get("type") == "repair"}
            reps = [r for r in reps if str(r.get("name")) in legal]
            if reps:
                best = max(reps, key=lambda r: int(r["maximum_amount"]))
                recs[name] = ({"type": "repair", "fault": best["name"],
                               "amount": int(best["maximum_amount"])},
                              f"standing on analysed fault {best['name']}: "
                              f"repair at max rate")
                continue
            if float(unit.get("remaining_resources") or 0) <= 0:
                recs[name] = ({"type": "stay"}, "out of repair resources")
                continue

        # 5) MPS island feeding, closure- and horizon-aware
        if utype == "MPS":
            plan = supply_plans.get(name) or {}
            supply = next((o for o in affordances["object_options"].get(name, [])
                           if o.get("type") == "supply"), None)
            if supply and not plan.get("grid_connected"):
                recs[name] = (supply, plan.get("reason", "island feed"))
                continue
            if plan.get("feasible_now") and (plan.get("island_unstable")
                                             or plan.get("blocked_by_repair")
                                             or not plan.get("sustainable")):
                recs[name] = ({"type": "continue"},
                              f"HOLD at {plan.get('node')}: {plan.get('reason','')}")
                continue

        # 6) movement
        target, why = None, ""

        if utype == "RC" and faults:
            for f in sorted(faults,
                            key=lambda f: _manhattan(here, _pos(f["position"]))):
                if f["name"] in claimed:
                    continue
                target = _clamp(_pos(f["position"]), topo)
                why = (f"nearest unclaimed fault {f['name']}"
                       f"{'' if f.get('analyzed') else ' (arrive -> 1 analysis step)'}")
                claimed.add(f["name"])
                break

        if target is None and utype in ("PAC", "RC") and unknown_px and topo is not None:
            route, covered = coverage_route(here, unknown_px, radius, topo,
                                            max_waypoints=5)
            if route:
                recs[name] = ({"type": "move", "waypoints": route},
                              f"patrol route revealing ~{covered} unknown "
                              f"network pixels")
                unknown_px = [p for p in unknown_px
                              if all(max(abs(p[0] - w[0]), abs(p[1] - w[1])) > radius
                                     for w in route)]
                continue

        if target is None and utype == "MPS" and unserved:
            cand = [(u["node"], node_pos[u["node"]], _load_value(u))
                    for u in unserved if u.get("node") in node_pos]
            cand = [c for c in cand if c[2] > 0]
            if cand:
                node, p, val = max(cand,
                                   key=lambda kv: kv[2] / (1 + _manhattan(here, kv[1])))
                target = _clamp(p, topo)
                why = f"park at node {node} (|P|*W={val:g}) to island-feed later"

        if target is None and centres:
            c = min(centres, key=lambda c: _manhattan(here, c))
            target, why = _clamp(c, topo), "patrol unknown region (fallback)"

        if target is not None and list(target) != list(here):
            recs[name] = ({"type": "move", "waypoints": [list(target)]}, why)
        else:
            recs[name] = ({"type": "continue"}, "nothing actionable")

    return recs


def recommended_command(state: dict, affordances: dict, topo=None) -> dict:
    """Full command batch from recommend(). This IS the scripted policy.

    switch_commands is intentionally empty: switching belongs to the
    environment-side SwitchOperator, which measures each toggle's true effect
    with /api/text/validate.
    """
    return {"switch_commands": {},
            "mobile_commands": {n: c for n, (c, _) in
                                recommend(state, affordances, topo).items()}}


# --------------------------------------------------------------------- #
# 3. agent-facing views
# --------------------------------------------------------------------- #
MISSION_BRIEF = """HOW SCORING WORKS (read carefully):
Score = sum over time of sum(|P| * W) over ENERGIZED loads. Restoring a
high-weight load EARLIER is worth more: it re-counts at every later step.
FATAL: if any load that was once energized loses power, the score is
permanently locked to zero for the rest of the episode.

Loads energize AUTOMATICALLY once their area is connected to a live source.
You cannot switch a load on. To make load energize you must:
  1. PATROL    - RC/PAC movement reveals unknown NODE and LINE pixels. Unknown
                 areas force switches open and block energization. Only
                 node/line pixels matter, not empty terrain.
  2. ANALYSE   - arriving at a known fault locks that unit for exactly ONE
                 step; send {"type":"continue"} for it that step.
  3. REPAIR    - the RC must be standing on the fault pixel, then repair.
  4. RECONNECT - close a switch into a KNOWN, FAULT-FREE area, or have an MPS
                 sit at a Node and supply the island's EXACT P and Q.

THE MPS IS THE #1 WAY TEAMS SCORE ZERO:
  (a) Starting a supply the MPS cannot hold to the final step. It burns P
      energy per step; when it runs out the simulator auto-disconnects it, its
      island goes dark, and the score locks.
  (b) Letting a supplying MPS run its energy down. Unrecoverable.
  (c) Supplying an island whose boundary is a FAULT rather than a SWITCH. When
      that fault is repaired the line re-closes, new load merges in, the
      islanded balance breaks, and the simulator then rejects EVERY command
      including a no-op. A supplying MPS cannot be re-dispatched.

Because of (c), an MPS is told to HOLD whenever its current island is smaller
than its FAULT CLOSURE -- the set of nodes reachable through switchless lines
if every fault were repaired. Repair first, then supply the larger STABLE
island. A short guaranteed window beats a long window that collapses.

NEVER command a Supplying MPS to move or stay: it disconnects immediately.
The transmission grid returns at its restoration step, but ONLY if the
source-connected network is fully known and fault-free by then. Once the grid
is connected it balances any MPS mismatch, so (c) only applies while islanded."""


def team_digest(state: dict, affordances: dict) -> str:
    game = state.get("game", {}) or {}
    summary = state.get("summary", {}) or {}
    t, n, remaining = horizon(state)

    L: List[str] = [
        f"T={t}/{n} ({remaining} steps remain)  score={game.get('score')}  "
        f"this-step |P|*W={game.get('weighted_restored_load')}  "
        f"locked={game.get('locked')}",
        f"grid={summary.get('source_status')} "
        f"(returns at T={summary.get('source_restore_step')})",
    ]

    term = affordances.get("terminal") or {}
    if term.get("terminal_fatal"):
        L.append("*** UNRECOVERABLE STATE: " + term.get("summary", "") + " ***")
    elif term.get("doomed_soon"):
        L.append("WARNING: " + term.get("summary", ""))

    fl = affordances["known_active_faults"]
    if fl:
        L.append("known active faults: " + ", ".join(
            f"{f['name']}@{f['position']}"
            + (f"[analysed, remaining={f.get('remaining_repair_resource')}]"
               if f.get("analyzed") else "[UNANALYSED]")
            for f in fl))
    else:
        L.append("known active faults: none discovered yet - PATROL to find them")

    L.append("units: " + "; ".join(
        f"{nm}({u.get('type')})@{u.get('position')} {u.get('state')}"
        f"{' LOCKED' if u.get('analysis_locked') else ''}"
        for nm, u in (state.get("objects") or {}).items()))

    for nm, plan in (affordances.get("supply_plans") or {}).items():
        if plan.get("island_changed"):
            L.append(f"{nm} ISLAND CHANGED (unrecoverable): outputs "
                     f"P={plan.get('current_p')} but the islanded cluster now "
                     f"needs P={plan.get('required_p')}")
        elif plan.get("supplying"):
            L.append(f"{nm} SUPPLYING P={plan.get('current_p')} "
                     f"(grid_connected={plan.get('grid_connected')}, "
                     f"{plan.get('steps_of_energy_left')} steps of energy)")
        elif plan.get("island_unstable"):
            extra = sorted(set(plan.get("closure_nodes") or [])
                           - set(plan.get("island_nodes") or []))
            L.append(f"{nm} SUPPLY ON HOLD (unstable island): cluster "
                     f"P={plan.get('p')} but closure P={plan.get('closure_p')}; "
                     f"{extra} are isolated only by a fault. Repair first.")
        elif plan.get("blocked_by_repair"):
            names = ", ".join(str(b.get("fault"))
                              for b in plan.get("blocking_faults") or [])
            L.append(f"{nm} SUPPLY ON HOLD (interlock): repairing [{names}] "
                     f"would grow this island")
        elif plan.get("feasible_now") and not plan.get("sustainable"):
            L.append(f"{nm} SUPPLY ON HOLD (energy): needs "
                     f"{plan.get('energy_needed'):g}; safe start step "
                     f"~{plan.get('start_step')}")
        elif plan.get("feasible_now"):
            L.append(f"{nm} SUPPLY IS NOW SAFE: P={plan['p']:g} Q={plan['q']:g} "
                     f"at {plan.get('node')} -- start it and never stop")

    flagged = affordances.get("blocked_repairs") or []
    if flagged:
        enforced = any(b.get("enforced") for b in flagged)
        verb = "BLOCKED" if enforced else "FLAGGED (advisory)"
        L.append(f"REPAIRS {verb}: " + ", ".join(
            f"{b['fault']} (would re-close {b['line']} and merge "
            f"{b['joins']} into {b['mps']}'s island)" for b in flagged))

    unsv = sorted(affordances["unenergized_loads"], key=lambda x: -_load_value(x))[:6]
    if unsv:
        L.append("top de-energized loads: " + ", ".join(
            f"{i['node']}(|P|*W={_load_value(i):g})" for i in unsv))

    ready = affordances.get("ready_areas") or []
    if ready:
        L.append("areas READY to energize (a boundary switch closure helps): " +
                 ", ".join(f"{a['area']}[{a['boundary_switches']}]" for a in ready))

    not_known = [a for a in (affordances.get("areas") or [])
                 if str(a.get("knowledge", "")).strip().lower() != "known"]
    if not_known:
        L.append("areas still not fully known: " + ", ".join(
            f"{a['area']}({a['knowledge']})" for a in not_known))

    L.append(f"unknown network pixels remaining: "
             f"{len(affordances.get('unknown_network_pixels') or [])}")
    return "\n".join(L)


def unit_block(state: dict, affordances: dict, name: str) -> str:
    u = (state.get("objects") or {}).get(name, {}) or {}
    L = [f"{name} type={u.get('type')} pos={u.get('position')} "
         f"state={u.get('state')} analysis_locked={u.get('analysis_locked')} "
         f"remaining_path_px={u.get('remaining_path_pixels')} "
         f"move_speed={u.get('move_speed')} inspection_range={u.get('inspection_range')}"]

    if u.get("type") == "RC":
        L.append(f"repair_speed={u.get('repair_speed')} "
                 f"resources={u.get('remaining_resources')}/{u.get('total_resources')}")
        legal = {o.get("fault") for o in
                 affordances["object_options"].get(name, [])
                 if o.get("type") == "repair"}
        for r in (u.get("repairable_faults") or []):
            if str(r.get("name")) in legal:
                L.append(f"  CAN REPAIR NOW: {r['name']} up to "
                         f"{r['maximum_amount']} this step")
            else:
                L.append(f"  REPAIR BLOCKED: {r['name']} would destabilise a "
                         f"supplying MPS and lock the score")

    if u.get("type") == "MPS":
        ctx = u.get("supply_context") or {}
        plan = (affordances.get("supply_plans") or {}).get(name) or {}
        L.append(f"energy={u.get('energy')} P<={u.get('p_limit')} "
                 f"S<={u.get('s_limit')} at_node={ctx.get('node')} "
                 f"cluster_P={ctx.get('load_p')} cluster_Q={ctx.get('load_q')} "
                 f"grid_connected={ctx.get('grid_connected')} "
                 f"cluster_safe={ctx.get('available')}")
        if plan:
            L.append(f"  SUPPLY ASSESSMENT: supplying={plan.get('supplying')} "
                     f"feasible={plan.get('feasible_now')} "
                     f"sustainable={plan.get('sustainable')} "
                     f"island_unstable={plan.get('island_unstable')} "
                     f"island_changed={plan.get('island_changed')} "
                     f"closure_P={plan.get('closure_p')} "
                     f"steps_of_energy_left={plan.get('steps_of_energy_left')} "
                     f":: {plan.get('reason')}")

    L.append("YOUR LEGAL COMMANDS (copy exactly one, verbatim):")
    for o in affordances["object_options"].get(name, []):
        L.append(f"  {json.dumps(o)}")

    mt = affordances["move_targets"].get(name) or []
    if mt:
        L.append("  legal move commands:")
        for t in mt:
            L.append(f'    {{"type":"move","waypoints":[{t["waypoint"]}]}}   <- {t["why"]}')
    return "\n".join(L)


def situation_card(state: dict, affordances: dict, topo=None) -> str:
    topo = topo or _TOPO
    recs = recommend(state, affordances, topo)

    L: List[str] = [MISSION_BRIEF, "", "SITUATION",
                    team_digest(state, affordances), ""]
    L.append("UNITS AND LEGAL COMMANDS:")
    for name in (state.get("objects") or {}):
        L.append(unit_block(state, affordances, name))
        L.append("")
    L.append(f"COMMANDABLE SWITCH TOGGLES: "
             f"{affordances['switch_toggle_options'] or '(none commandable)'}")
    L.append("NOTE: switch closures are chosen automatically by the "
             "SwitchOperator and validated against the simulator. Leave "
             "switch_commands empty.")
    L += ["", "RECOMMENDED ACTION THIS STEP (strong prior - follow unless you "
              "have a concrete, stated better reason):"]
    for name, (cmd, why) in recs.items():
        L.append(f"  {name}: {json.dumps(cmd)}   <- {why}")
    return "\n".join(L)