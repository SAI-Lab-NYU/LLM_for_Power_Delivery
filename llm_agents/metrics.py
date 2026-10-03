"""Dense progress potential + communication/collaboration metrics.

WHY a potential: the transmission grid returns only at step 6 and the unknown
region is large, so d(weighted restored load) == 0 for most of the episode.
Ranking candidates on delta alone is blind exactly when the decisions matter.

FIX IN THIS VERSION -- Phi MONOTONICITY
---------------------------------------
Observed Phi trace from a real run:  26 -> 29 -> 35 -> 17 -> -6 -> -12 -> 27
Phi DROPPED when the team made progress. Cause: an unanalysed fault reports
remaining_repair_resource = null, which we read as 0. The moment a crew
analyses it, the true workload becomes visible and is weighted -3x, so
*discovering work looks like regression*.

That corrupts (a) the SwitchOperator gain signal and (b) candidate ranking in
agents.py -- precisely the two places Phi is used.

Fix: charge every known fault an EXPECTED workload until it is analysed. Phi
then rises on analysis (information gained) instead of collapsing.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- potential weights ---------------------------------------------- #
W_REPAIR = 3.0          # per unit of remaining repair work (negative)
W_KNOWN = 1.0           # per known node/line element
W_ENERGIZABLE = 40.0    # per area that is ready to energize
W_ANALYZED = 15.0       # per analysed active fault (information bonus)
W_FAULT_FOUND = 5.0     # per discovered fault (discovery bonus)

# Assumed workload of a known-but-unanalysed fault. Scenario generator draws
# repair_resource from [5, 14], so 10 is the midpoint.
EXPECTED_UNANALYSED_WORK = 10.0


# --------------------------------------------------------------------- #
# potential
# --------------------------------------------------------------------- #
def potential(state: Dict[str, Any],
              affordances: Optional[Dict[str, Any]] = None) -> float:
    """Dense, MONOTONE progress potential.

    Monotone means: revealing information, analysing a fault, repairing work,
    and making an area energizable all move Phi UP (or leave it flat). Nothing
    the team can legitimately do makes Phi fall.
    """
    faults = [f for f in (state.get("faults") or []) if f.get("active")]

    remaining = 0.0
    analyzed = 0
    for f in faults:
        if f.get("analyzed"):
            analyzed += 1
            try:
                remaining += float(f.get("remaining_repair_resource") or 0.0)
            except (TypeError, ValueError):
                pass
        else:
            # FIX: charge an expected workload so that analysing a fault does
            # not suddenly add -3 * workload and crater Phi.
            remaining += EXPECTED_UNANALYSED_WORK

    rows = (((state.get("status_tables") or {}).get("elements") or {}).get("rows")) or []
    known = sum(1 for r in rows
                if len(r) > 2 and r[0] in ("Node", "Line")
                and str(r[2]).strip().lower() == "known")

    arows = (((state.get("status_tables") or {}).get("areas") or {}).get("rows")) or []
    energizable = sum(1 for r in arows
                      if len(r) > 5 and str(r[5]).strip().lower() == "yes")

    return (-W_REPAIR * remaining
            + W_ANALYZED * analyzed
            + W_FAULT_FOUND * len(faults)
            + W_KNOWN * known
            + W_ENERGIZABLE * energizable)


def potential_delta(predicted_state: Dict[str, Any], base: float) -> float:
    if not predicted_state:
        return 0.0
    return potential(predicted_state) - base


def potential_breakdown(state: Dict[str, Any]) -> Dict[str, float]:
    """Component-wise Phi, for debugging a surprising value."""
    faults = [f for f in (state.get("faults") or []) if f.get("active")]
    remaining = 0.0
    analyzed = 0
    for f in faults:
        if f.get("analyzed"):
            analyzed += 1
            try:
                remaining += float(f.get("remaining_repair_resource") or 0.0)
            except (TypeError, ValueError):
                pass
        else:
            remaining += EXPECTED_UNANALYSED_WORK
    rows = (((state.get("status_tables") or {}).get("elements") or {}).get("rows")) or []
    known = sum(1 for r in rows if len(r) > 2 and r[0] in ("Node", "Line")
                and str(r[2]).strip().lower() == "known")
    arows = (((state.get("status_tables") or {}).get("areas") or {}).get("rows")) or []
    energizable = sum(1 for r in arows if len(r) > 5
                      and str(r[5]).strip().lower() == "yes")
    return {
        "remaining_work": remaining,
        "repair_term": -W_REPAIR * remaining,
        "analyzed": analyzed,
        "analyzed_term": W_ANALYZED * analyzed,
        "faults_found": len(faults),
        "found_term": W_FAULT_FOUND * len(faults),
        "known_elements": known,
        "known_term": W_KNOWN * known,
        "energizable_areas": energizable,
        "energizable_term": W_ENERGIZABLE * energizable,
        "total": potential(state),
    }


# --------------------------------------------------------------------- #
# fault accounting  (FIX for the bogus `faults_repaired` column)
# --------------------------------------------------------------------- #
def count_faults(state: Dict[str, Any],
                 scenario_faults: Optional[List[dict]] = None) -> Dict[str, int]:
    """Honest fault accounting.

    The text state exposes ONLY known faults, so "not in the list" is
    ambiguous: it can mean repaired OR never discovered. The previous code
    treated it as repaired, which is why `noop` reported "repaired 1/3" while
    issuing zero commands.

    We therefore use the scenario's ground-truth fault list as the denominator
    and count a fault repaired only if it IS present and inactive.
    """
    listed = state.get("faults") or []
    by_name = {str(f.get("name")): f for f in listed}

    total = len(scenario_faults) if scenario_faults else len(listed)
    known = len(listed)
    active_known = sum(1 for f in listed if f.get("active"))
    analyzed = sum(1 for f in listed if f.get("active") and f.get("analyzed"))
    repaired = sum(1 for f in listed if not f.get("active"))

    if scenario_faults:
        undiscovered = max(0, total - known)
    else:
        undiscovered = 0

    return {
        "total": total,
        "known": known,
        "undiscovered": undiscovered,
        "active_known": active_known,
        "analyzed": analyzed,
        "repaired": repaired,
        # anything not discovered is definitionally NOT repaired
        "unresolved": active_known + undiscovered,
    }


# --------------------------------------------------------------------- #
# communication metrics (from a blackboard transcript)
# --------------------------------------------------------------------- #
def _load(path: Path) -> List[dict]:
    out: List[dict] = []
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return out


def communication_metrics(transcript_path: Path) -> Dict[str, Any]:
    """Cost + quality of the inter-agent dialogue."""
    msgs = _load(Path(transcript_path))
    if not msgs:
        return {}

    by_kind = Counter(str(m.get("kind")) for m in msgs)
    by_speaker = Counter(str(m.get("speaker")) for m in msgs)
    llm = [m for m in msgs if m.get("raw")]
    parse_fail = sum(1 for m in llm if not m.get("parse_ok", True))
    steps = sorted({int(m.get("t", 0)) for m in msgs})

    # proposal survival: did a unit's round-1 command reach the executed batch?
    survived = kept = 0
    for t in steps:
        props = {str(m["speaker"]): (m.get("payload") or {}).get("command")
                 for m in msgs
                 if int(m.get("t", 0)) == t and m.get("kind") == "proposal"}
        final = None
        for m in reversed(msgs):
            if int(m.get("t", 0)) == t and m.get("kind") == "assignment":
                final = (m.get("payload") or {}).get("mobile_commands")
                break
        final = final or {}
        for name, cmd in props.items():
            if cmd is None:
                continue
            survived += 1
            if json.dumps(final.get(name), sort_keys=True) == json.dumps(cmd, sort_keys=True):
                kept += 1

    verdicts = [m for m in msgs if m.get("kind") == "verdict"]
    rejects = sum(1 for m in verdicts
                  if str((m.get("payload") or {}).get("verdict")) == "reject")

    # conflicts detected vs. resolved by talking
    conf_before = conf_after = 0
    for m in msgs:
        pay = m.get("payload") or {}
        if "n_conflicts" in pay:
            conf_before += int(pay.get("n_conflicts") or 0)
            conf_after += int(pay.get("n_conflicts_after") or 0)

    influenced = sum(1 for m in msgs
                     if (m.get("payload") or {}).get("influenced_by"))
    yields = sum(1 for m in msgs
                 if (m.get("payload") or {}).get("yielded"))

    return {
        "steps": len(steps),
        "messages": len(msgs),
        "messages_per_step": round(len(msgs) / max(len(steps), 1), 2),
        "llm_calls": len(llm),
        "llm_calls_per_step": round(len(llm) / max(len(steps), 1), 2),
        "mean_latency_s": round(sum(float(m.get("latency_s") or 0) for m in llm)
                                / max(len(llm), 1), 2),
        "total_response_chars": sum(int(m.get("response_chars") or 0) for m in llm),
        "schema_fail_rate": round(parse_fail / max(len(llm), 1), 3),
        "proposal_survival_rate": round(kept / max(survived, 1), 3),
        "requests_sent": by_kind.get("request", 0),
        "objections": by_kind.get("objection", 0),
        "reflections": by_kind.get("reflection", 0),
        "verdict_reject_rate": round(rejects / max(len(verdicts), 1), 3),
        "conflicts_detected": conf_before,
        "conflicts_remaining": conf_after,
        "conflict_resolution_rate": round(
            1.0 - (conf_after / conf_before), 3) if conf_before else None,
        "peer_influenced_proposals": influenced,
        "yields": yields,
        "by_kind": dict(by_kind),
        "by_speaker": dict(by_speaker),
    }