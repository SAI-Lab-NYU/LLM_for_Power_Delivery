"""Render the planner -> unit DISPATCH CHAIN for single-agent arms.

WHY THIS IS NOT "COMMUNICATION"
-------------------------------
LARA is architecturally a single planner: it emits ONE joint macro-action
covering every unit (paper Sec. 2.5). There are no proposals, requests, or
objections between units, so there is no inter-agent dialogue to render.
conversation.py correctly reports collaboration_score ~25 and
listening_rate 0 for this arm, and that number should be read as N/A.

What this module renders instead is the command trace that actually exists:

    affordance catalogue -> k sampled plans -> rollout scores -> chosen plan
      -> per-unit orders -> SwitchOperator report -> environment response

This is the interpretability artefact LARA claims as its differentiator over
MILP and CTDE-RL: a human can read why each crew was sent where, and patch it.

INPUTS -- existing logs only. No simulator, no GPU, no re-run.
  interactions_<tag>.jsonl : stage=="lara" records (plans, rollout scores,
                             chosen_index, reasoning, switch_report, latency)
  trace_<tag>.json         : per-step executed batch + environment response

Also works for scripted / scripted_switch / gen_* (those have no planner
reasoning, so only the ORDERS and ENV blocks are populated).

Usage
  python -m llm_agents.dispatch llm_agents/results/lara_repro/trace_lara_ep0_t0.json
  python -m llm_agents.dispatch llm_agents/results/lara_repro --method lara --out dispatch.md
  python -m llm_agents.dispatch <dir> --summary-only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

UNIT_ICON = {"RC": "[RC ]", "PAC": "[PAC]", "MPS": "[MPS]"}


# --------------------------------------------------------------------- #
def load_trace(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_interactions(path: Path) -> List[dict]:
    out: List[dict] = []
    if not Path(path).exists():
        return out
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def tag_from_trace(path: Path) -> str:
    name = Path(path).stem
    return name[len("trace_"):] if name.startswith("trace_") else name


def interactions_for(trace_path: Path) -> List[dict]:
    tag = tag_from_trace(trace_path)
    return load_interactions(Path(trace_path).parent / f"interactions_{tag}.jsonl")


# --------------------------------------------------------------------- #
def order_text(name: str, cmd: Any) -> str:
    """One line of plain English for a single unit's order."""
    if not isinstance(cmd, dict):
        return f"{name}: {cmd}"
    t = str(cmd.get("type", "continue")).strip().lower()
    if t == "repaire":
        t = "repair"
    if t == "move":
        wps = cmd.get("waypoints") or []
        pts = " -> ".join(f"({int(p[0])},{int(p[1])})" for p in wps
                          if isinstance(p, (list, tuple)) and len(p) >= 2)
        return f"{name}: MOVE via {pts or '(none)'}"
    if t == "repair":
        return (f"{name}: REPAIR fault {cmd.get('fault')} "
                f"for {cmd.get('amount')} units this step")
    if t == "supply":
        return (f"{name}: SUPPLY island with P={cmd.get('p')} kW, "
                f"Q={cmd.get('q')} kVar  (must hold to the final step)")
    if t == "stay":
        return f"{name}: STAY (go idle here)"
    return f"{name}: CONTINUE current activity"


def unit_type_of(name: str) -> str:
    n = name.upper()
    for k in ("MPS", "PAC", "RC"):
        if n.startswith(k):
            return k
    return "RC"


def render_orders(batch: Dict[str, Any], indent: str = "     ") -> List[str]:
    out: List[str] = []
    mob = (batch or {}).get("mobile_commands") or {}
    for name in sorted(mob):
        icon = UNIT_ICON.get(unit_type_of(name), "[?  ]")
        out.append(f"{indent}{icon} {order_text(name, mob[name])}")
    sw = (batch or {}).get("switch_commands") or {}
    out.append(f"{indent}[SW ] switches: {sw if sw else '(none this step)'}")
    return out


def _num(x, nd: int = 0) -> str:
    try:
        if x is None:
            return "-"
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


# --------------------------------------------------------------------- #
def render_dispatch(trace: Dict[str, Any], inter: List[dict],
                    steps: Optional[List[int]] = None,
                    width: int = 84) -> str:
    summary = trace.get("summary") or {}
    scen = trace.get("scenario") or {}
    rows = trace.get("trace") or []
    lara_by_t: Dict[int, dict] = {int(r.get("t", -1)): r for r in inter
                                  if r.get("stage") == "lara"}

    L: List[str] = []
    L.append("=" * width)
    L.append(f"DISPATCH CHAIN  --  {summary.get('method')} "
             f"ep{summary.get('episode')} trial{summary.get('trial')}")
    L.append("=" * width)
    L.append(f"model        : {summary.get('model')}")
    L.append(f"scenario     : seed={scen.get('seed')} "
             f"faults={len(scen.get('faults') or [])} "
             f"work={scen.get('total_repair')}")
    for f in (scen.get("faults") or []):
        L.append(f"               fault {f.get('name')} at "
                 f"{f.get('position')} needs {f.get('repair_resource')} units")
    L.append(f"outcome      : score={_num(summary.get('final_score'))} "
             f"steps={summary.get('steps_taken')} "
             f"locked={summary.get('locked')} "
             f"repaired={summary.get('faults_repaired')}/"
             f"{summary.get('n_faults')}")
    if summary.get("decision_points") is not None:
        L.append(f"planner      : {summary.get('decision_points')} decision "
                 f"points over {summary.get('steps_taken')} steps; "
                 f"median latency "
                 f"{summary.get('median_decision_latency_s')}s; "
                 f"{summary.get('llm_calls')} LLM calls")
    L.append("")
    L.append("NOTE: this arm is a SINGLE planner. The blocks below are a")
    L.append("      command trace, not inter-agent communication.")
    L.append("")

    prev_score = 0.0
    for row in rows:
        t = int(row.get("t", 0))
        if steps and t not in steps:
            prev_score = float(row.get("score_after") or prev_score)
            continue

        rec = lara_by_t.get(t)
        replanned = bool(row.get("replanned")) if "replanned" in row else (rec is not None)
        header = (f"TIME STEP {t}        "
                  f"{'DECISION POINT (replanned)' if replanned else 'executing committed plan'}")
        L.append("-" * width)
        L.append(header)
        L.append("-" * width)

        # ---- catalogue + planner reasoning -------------------------
        if rec:
            L.append(f"  [CATALOGUE] k={rec.get('n_sampled')} sampled -> "
                     f"{rec.get('n_plans')} unique after dedup "
                     f"({rec.get('n_viable')} legal & non-fatal)")
            reasoning = str(rec.get("reasoning") or "").strip()
            if reasoning:
                L.append("  [PLANNER]   " + reasoning[:width - 14])
                for extra in [reasoning[i:i + width - 14]
                              for i in range(width - 14, len(reasoning), width - 14)]:
                    L.append("              " + extra)
            L.append(f"  [LATENCY]   {rec.get('latency_s')}s at this decision point")

            # ---- candidate plans ----------------------------------
            plans = rec.get("plans") or []
            if plans:
                L.append("  [CANDIDATES]")
                chosen = rec.get("chosen_index")
                for p in plans:
                    mark = "  <== CHOSEN" if p.get("i") == chosen else ""
                    status = ("ok" if p.get("sim_ok") else "REJECTED BY SIM")
                    if p.get("fatal"):
                        status = "FATAL (loses restored load)"
                    L.append(f"     plan {p.get('i')}  "
                             f"score={_num(p.get('score'), 1)}  "
                             f"dW={_num(p.get('dW'))}  "
                             f"dPhi={_num(p.get('dPhi'), 1)}  "
                             f"[{status}]{mark}")
                    if not p.get("parse_ok", True):
                        L.append("        !! model output failed to parse")
                    if p.get("msg"):
                        L.append(f"        sim: {str(p['msg'])[:width - 14]}")
                    L.extend(render_orders(p.get("cmd") or {}, indent="        "))
            if rec.get("switch_report"):
                L.append("  [SWITCHES]  " +
                         "; ".join(str(s) for s in rec["switch_report"])[:width * 3])
        elif row.get("switch_report"):
            L.append("  [SWITCHES]  " +
                     "; ".join(str(s) for s in row["switch_report"])[:width * 3])

        # ---- executed orders ---------------------------------------
        L.append("  [ORDERS EXECUTED]")
        L.extend(render_orders(row.get("command") or {}))

        # ---- environment response ----------------------------------
        score_after = float(row.get("score_after") or 0.0)
        L.append(f"  [ENV]       "
                 f"{'accepted' if row.get('executed_ok') else 'REJECTED'}"
                 f" | score {_num(prev_score)} -> {_num(score_after)}"
                 f" | this-step |P|*W={_num(row.get('weighted_after'))}"
                 f" | locked={row.get('locked_after')}")
        if row.get("message") and not row.get("executed_ok"):
            L.append(f"              sim says: {str(row['message'])[:width - 14]}")
        if row.get("violation_loads"):
            L.append(f"              LOST LOADS: {row['violation_loads']}")
        if row.get("illegal_commands"):
            L.append(f"              illegal commands in batch: "
                     f"{row['illegal_commands']}")
        if row.get("terminal_fatal"):
            L.append("              *** UNRECOVERABLE STATE ***")
        prev_score = score_after
        L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------- #
def dispatch_stats(trace: Dict[str, Any], inter: List[dict]) -> Dict[str, Any]:
    """Quantify the dispatch chain: plan diversity and where the LLM mattered."""
    rows = trace.get("trace") or []
    recs = [r for r in inter if r.get("stage") == "lara"]
    chosen = Counter(r.get("chosen_index") for r in recs)
    kept = Counter(int(r.get("n_plans") or 0) for r in recs)
    sampled = sum(int(r.get("n_sampled") or 0) for r in recs)
    uniq = sum(int(r.get("n_plans") or 0) for r in recs)

    # distribution of order types actually executed
    order_types: Counter = Counter()
    for row in rows:
        for name, cmd in ((row.get("command") or {})
                          .get("mobile_commands") or {}).items():
            t = str((cmd or {}).get("type", "continue")).lower()
            order_types[f"{unit_type_of(name)}:{'repair' if t == 'repaire' else t}"] += 1

    scores = [p.get("score") for r in recs for p in (r.get("plans") or [])
              if isinstance(p.get("score"), (int, float))]
    discriminable = sum(1 for r in recs
                        if len({round(float(p.get("score") or 0), 3)
                                for p in (r.get("plans") or [])
                                if isinstance(p.get("score"), (int, float))}) > 1)
    return {
        "steps": len(rows),
        "decision_points": len(recs),
        "replan_rate": round(len(recs) / max(len(rows), 1), 3),
        "plans_sampled": sampled,
        "plans_unique": uniq,
        "dedup_rate": round(1.0 - uniq / max(sampled, 1), 3),
        "kept_histogram": dict(sorted(kept.items())),
        "chosen_index_histogram": {str(k): v for k, v in sorted(
            chosen.items(), key=lambda kv: (kv[0] is None, kv[0]))},
        "decision_points_with_distinct_scores": discriminable,
        "discriminable_rate": round(discriminable / max(len(recs), 1), 3),
        "max_rollout_score": max(scores) if scores else None,
        "median_abs_rollout_score": (
            sorted(abs(s) for s in scores)[len(scores) // 2] if scores else None),
        "order_types": dict(order_types.most_common()),
    }


def render_stats(st: Dict[str, Any], width: int = 84) -> str:
    L = ["=" * width, "DISPATCH STATISTICS", "=" * width]
    L.append(f"steps={st['steps']}  decision_points={st['decision_points']}  "
             f"replan_rate={st['replan_rate']}")
    L.append(f"plans sampled={st['plans_sampled']} -> unique={st['plans_unique']}  "
             f"(dedup rate {st['dedup_rate']})")
    L.append(f"kept-per-decision histogram : {st['kept_histogram']}")
    L.append(f"chosen plan index histogram : {st['chosen_index_histogram']}")
    L.append(f"decision points where candidates had DISTINCT rollout scores: "
             f"{st['decision_points_with_distinct_scores']}"
             f"/{st['decision_points']} ({st['discriminable_rate']})")
    L.append(f"median |rollout score| = {st['median_abs_rollout_score']}  "
             f"max = {st['max_rollout_score']}")
    L.append(f"executed order types: {st['order_types']}")
    L.append("")
    L.append("How to read this:")
    L.append("  chosen index histogram concentrated on 0 + low discriminable")
    L.append("  rate => the selector had nothing to choose between, so k is")
    L.append("  buying nothing. Raise temperature or weaken the RECOMMENDED")
    L.append("  ACTION block in the prompt.")
    L.append("=" * width)
    return "\n".join(L)


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="trace_*.json, or a results directory")
    ap.add_argument("--method", default=None,
                    help="when PATH is a directory, filter traces by method")
    ap.add_argument("--steps", nargs="*", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        pattern = f"trace_{args.method}_ep*.json" if args.method else "trace_*.json"
        files = sorted(p.glob(pattern))
    else:
        files = [p]
    if not files:
        print(f"No traces found at {p}")
        return

    chunks: List[str] = []
    for f in files:
        trace = load_trace(f)
        inter = interactions_for(f)
        st = dispatch_stats(trace, inter)
        parts = [f"\n### {f.name}\n"]
        if not args.summary_only:
            parts.append(render_dispatch(trace, inter, args.steps))
        parts.append(render_stats(st))
        chunks.append("\n".join(parts))

    text = "\n".join(chunks)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}  ({len(text):,} chars)")
    else:
        print(text)


if __name__ == "__main__":
    main()