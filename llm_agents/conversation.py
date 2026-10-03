"""Render the blackboard as a readable dialogue, and MEASURE collaboration.

Two questions this answers:
  1. "What did the agents actually say to each other?"      -> render_dialogue()
  2. "Are they genuinely collaborating, or just talking?"   -> collaboration_report()

The distinction matters. An agent that emits a polite thought every step but
never changes its action because of a teammate is NOT collaborating; it is
narrating. We therefore measure influence, not volume:

  * peer_influence_rate  - proposals that cite a teammate message AND differ
                           from what the agent would otherwise have done
  * yield_rate           - conflicts where an agent explicitly gave up a target
  * conflict_resolution  - conflicts resolved by dialogue vs. by the
                           deterministic tie-breaker
  * listening_rate       - proposals referencing any peer message at all
  * reciprocity          - fraction of directed requests that got a response

Usage
  python -m llm_agents.conversation llm_agents/results/multi/transcript_ma_comm_ep0_t0.jsonl
  python -m llm_agents.conversation <jsonl> --out dialogue.md --raw
  python -m llm_agents.conversation <dir>  --report-only
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROUND_NAMES = {
    0: "ROUND 0 - Dispatcher briefing",
    1: "ROUND 1 - Unit proposals",
    2: "ROUND 2 - Negotiation",
    3: "ROUND 3 - Assembly + switching",
    4: "ROUND 4 - Safety review",
    5: "ROUND 5 - Execution + reflection",
}

KIND_ORDER = {"digest": 0, "proposal": 1, "request": 2, "reply": 3,
              "objection": 4, "yield": 5, "switchplan": 6, "assignment": 7,
              "verdict": 8, "execution": 9, "reflection": 10, "patch": 11}

ICON = {"digest": "[BRIEF]", "proposal": "[PROPOSE]", "request": "[ASK]",
        "reply": "[REPLY]", "objection": "[OBJECT]", "yield": "[YIELD]",
        "switchplan": "[SWITCH]", "assignment": "[ASSIGN]",
        "verdict": "[JUDGE]", "execution": "[ENV]",
        "reflection": "[REFLECT]", "patch": "[HUMAN]"}

UNIT_ROLES = {"RC": "repair crew", "PAC": "patrol crew", "MPS": "mobile power"}


# --------------------------------------------------------------------- #
def load(path: Path) -> List[dict]:
    out: List[dict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _cmd_str(cmd: Any) -> str:
    if not isinstance(cmd, dict):
        return str(cmd)
    t = str(cmd.get("type", "?"))
    if t == "move":
        wps = cmd.get("waypoints") or []
        return f"move -> {wps}"
    if t in ("repair", "repaire"):
        return f"repair {cmd.get('fault')} x{cmd.get('amount')}"
    if t == "supply":
        return f"supply P={cmd.get('p')} Q={cmd.get('q')}"
    return t


def _sort_key(m: dict) -> Tuple:
    return (int(m.get("round", 0)),
            KIND_ORDER.get(str(m.get("kind")), 99),
            float(m.get("ts", 0.0)))


# --------------------------------------------------------------------- #
# 1. dialogue rendering
# --------------------------------------------------------------------- #
def render_dialogue(msgs: List[dict], steps: Optional[List[int]] = None,
                    include_raw: bool = False, width: int = 100) -> str:
    """Human-readable transcript, grouped by step then dialogue round."""
    all_steps = sorted({int(m.get("t", 0)) for m in msgs})
    show = steps if steps else all_steps
    lines: List[str] = []

    for t in show:
        step = sorted([m for m in msgs if int(m.get("t", 0)) == t], key=_sort_key)
        if not step:
            continue
        lines.append("=" * width)
        lines.append(f"TIME STEP {t}")
        lines.append("=" * width)

        cur_round = None
        for m in step:
            r = int(m.get("round", 0))
            if r != cur_round:
                cur_round = r
                lines.append("")
                lines.append(f"--- {ROUND_NAMES.get(r, f'ROUND {r}')} ---")

            kind = str(m.get("kind"))
            spk = str(m.get("speaker"))
            to = m.get("to") or ["ALL"]
            tag = f"{ICON.get(kind, '[' + kind.upper() + ']'):<10} {spk}"
            if to != ["ALL"]:
                tag += f" -> {', '.join(str(x) for x in to)}"
            lines.append(tag)

            text = str(m.get("text") or "").strip()
            if text:
                for chunk in text.split("\n"):
                    chunk = chunk.strip()
                    while len(chunk) > width - 6:
                        cut = chunk.rfind(" ", 0, width - 6)
                        cut = cut if cut > 0 else width - 6
                        lines.append(f'     "{chunk[:cut]}"')
                        chunk = chunk[cut:].strip()
                    if chunk:
                        lines.append(f'     "{chunk}"')

            pay = m.get("payload") or {}
            if kind in ("proposal", "objection", "yield") and "command" in pay:
                lines.append(f"     ACTION: {_cmd_str(pay['command'])}"
                             + (f"  (confidence {pay['confidence']})"
                                if "confidence" in pay else ""))
                if pay.get("influenced_by"):
                    lines.append(f"     >> influenced by {pay['influenced_by']}")
                if pay.get("yielded"):
                    lines.append(f"     >> YIELDED {pay['yielded']}")
            elif kind == "digest" and pay.get("intents"):
                for u, intent in pay["intents"].items():
                    if intent:
                        lines.append(f"     {u}: {intent}")
            elif kind == "assignment":
                mob = pay.get("mobile_commands") or {}
                sw = pay.get("switch_commands") or {}
                for u, c in mob.items():
                    lines.append(f"     {u}: {_cmd_str(c)}")
                lines.append(f"     switches: {sw if sw else '(none)'}")
                if pay.get("n_conflicts") is not None:
                    lines.append(f"     conflicts: {pay.get('n_conflicts')} "
                                 f"-> {pay.get('n_conflicts_after')} after talking")
            elif kind == "verdict":
                lines.append(f"     verdict={pay.get('verdict')} "
                             f"risk={pay.get('risk')} score={pay.get('score')}")
                if pay.get("suggestion"):
                    lines.append(f"     suggestion: {pay['suggestion']}")
            elif kind == "execution":
                lines.append(f"     score={pay.get('score')} "
                             f"|P|*W={pay.get('weighted')} "
                             f"locked={pay.get('locked')}")
                en = pay.get("energized")
                if en is not None:
                    lines.append(f"     energized: {en}")
            elif kind == "switchplan":
                sw = pay.get("switch_commands") or {}
                lines.append(f"     toggles: {sw if sw else '(none)'}")
            elif pay:
                blob = json.dumps(pay, default=str)
                lines.append(f"     {blob[:width - 6]}"
                             + ("..." if len(blob) > width - 6 else ""))

            if m.get("parse_ok") is False:
                lines.append("     !! MODEL OUTPUT FAILED TO PARSE")
            if include_raw and m.get("raw"):
                lines.append("     RAW: " + str(m["raw"])[:500].replace("\n", " "))
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- #
# 2. collaboration report
# --------------------------------------------------------------------- #
def collaboration_report(msgs: List[dict]) -> Dict[str, Any]:
    """Is this genuine collaboration, or parallel monologue?"""
    if not msgs:
        return {}

    steps = sorted({int(m.get("t", 0)) for m in msgs})
    proposals = [m for m in msgs if m.get("kind") == "proposal"]
    requests = [m for m in msgs if m.get("kind") == "request"]
    replies = [m for m in msgs if m.get("kind") == "reply"]
    objections = [m for m in msgs if m.get("kind") == "objection"]
    yields_ = [m for m in msgs if m.get("kind") == "yield"
               or (m.get("payload") or {}).get("yielded")]

    units = sorted({str(m["speaker"]) for m in proposals})

    # --- who talks, who listens ------------------------------------
    per_unit: Dict[str, Dict[str, Any]] = {}
    for u in units:
        mine = [m for m in proposals if str(m["speaker"]) == u]
        influenced = [m for m in mine if (m.get("payload") or {}).get("influenced_by")]
        my_reqs = [m for m in requests if str(m["speaker"]) == u]
        to_me = [m for m in requests if u in (m.get("to") or [])]
        my_replies = [m for m in replies if str(m["speaker"]) == u]
        per_unit[u] = {
            "proposals": len(mine),
            "requests_sent": len(my_reqs),
            "requests_received": len(to_me),
            "replies_sent": len(my_replies),
            "objections": sum(1 for m in objections if str(m["speaker"]) == u),
            "yields": sum(1 for m in yields_ if str(m["speaker"]) == u),
            "peer_influenced": len(influenced),
            "peer_influence_rate": round(len(influenced) / max(len(mine), 1), 3),
            "parse_failures": sum(1 for m in mine if m.get("parse_ok") is False),
            "mean_confidence": round(
                sum(float((m.get("payload") or {}).get("confidence") or 0)
                    for m in mine) / max(len(mine), 1), 2),
        }

    # --- directed message graph ------------------------------------
    graph: Dict[str, Counter] = defaultdict(Counter)
    for m in requests + replies + objections:
        for tgt in (m.get("to") or []):
            if tgt != "ALL":
                graph[str(m["speaker"])][str(tgt)] += 1

    # --- reciprocity: did a directed request get any answer? -------
    answered = 0
    for req in requests:
        spk = str(req["speaker"])
        tgts = [x for x in (req.get("to") or []) if x != "ALL"]
        t = int(req.get("t", 0))
        for m in msgs:
            if int(m.get("t", 0)) != t:
                continue
            if str(m.get("speaker")) not in tgts:
                continue
            if m.get("kind") in ("reply", "objection", "yield"):
                answered += 1
                break
            if m.get("kind") == "proposal" and \
               spk in str((m.get("payload") or {}).get("influenced_by") or ""):
                answered += 1
                break

    # --- conflicts: detected vs. talked out ------------------------
    conf_before = conf_after = 0
    for m in msgs:
        pay = m.get("payload") or {}
        if "n_conflicts" in pay:
            conf_before += int(pay.get("n_conflicts") or 0)
            conf_after += int(pay.get("n_conflicts_after") or 0)

    # --- proposal survival ----------------------------------------
    survived = kept = 0
    for t in steps:
        props = {str(m["speaker"]): (m.get("payload") or {}).get("command")
                 for m in msgs if int(m.get("t", 0)) == t
                 and m.get("kind") == "proposal"}
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
            if json.dumps(final.get(name), sort_keys=True) == \
               json.dumps(cmd, sort_keys=True):
                kept += 1

    n_steps = max(len(steps), 1)
    n_prop = max(len(proposals), 1)
    listening = sum(1 for m in proposals
                    if (m.get("payload") or {}).get("influenced_by"))

    # --- composite collaboration score (0-100) --------------------
    # Equal weight on: are they talking, are they listening, do they resolve
    # conflicts, and is dialogue bidirectional.
    talk = min(1.0, (len(requests) + len(replies) + len(objections)) / (2.0 * n_steps))
    listen = listening / n_prop
    resolve = (1.0 - conf_after / conf_before) if conf_before else 1.0
    recip = answered / max(len(requests), 1)
    score = round(100.0 * (0.25 * talk + 0.30 * listen
                           + 0.25 * resolve + 0.20 * recip), 1)

    if score >= 60:
        label = "collaborating"
    elif score >= 30:
        label = "partially collaborating"
    elif len(requests) + len(objections) > 0:
        label = "talking but not adapting"
    else:
        label = "parallel monologue (no real collaboration)"

    return {
        "steps": len(steps),
        "units": units,
        "messages": len(msgs),
        "messages_per_step": round(len(msgs) / n_steps, 2),
        "proposals": len(proposals),
        "requests": len(requests),
        "replies": len(replies),
        "objections": len(objections),
        "yields": len(yields_),
        "peer_influenced_proposals": listening,
        "listening_rate": round(listen, 3),
        "reciprocity_rate": round(recip, 3),
        "conflicts_detected": conf_before,
        "conflicts_remaining": conf_after,
        "conflict_resolution_rate": round(resolve, 3) if conf_before else None,
        "proposal_survival_rate": round(kept / max(survived, 1), 3),
        "collaboration_score": score,
        "collaboration_verdict": label,
        "per_unit": per_unit,
        "message_graph": {k: dict(v) for k, v in graph.items()},
    }


def render_report(rep: Dict[str, Any], width: int = 100) -> str:
    if not rep:
        return "(no messages)"
    L = ["=" * width, "COLLABORATION REPORT", "=" * width,
         f"verdict: {rep['collaboration_verdict'].upper()}  "
         f"(score {rep['collaboration_score']}/100)", "",
         f"steps={rep['steps']}  messages={rep['messages']} "
         f"({rep['messages_per_step']}/step)  units={', '.join(rep['units'])}", ""]

    L.append("Dialogue volume")
    L.append(f"  proposals={rep['proposals']}  requests={rep['requests']}  "
             f"replies={rep['replies']}  objections={rep['objections']}  "
             f"yields={rep['yields']}")
    L.append("")
    L.append("Did they actually influence each other?")
    L.append(f"  peer-influenced proposals : {rep['peer_influenced_proposals']}"
             f"/{rep['proposals']}  (listening rate {rep['listening_rate']})")
    L.append(f"  reciprocity (asks answered): {rep['reciprocity_rate']}")
    crr = rep.get("conflict_resolution_rate")
    L.append(f"  conflicts {rep['conflicts_detected']} -> "
             f"{rep['conflicts_remaining']}"
             + (f"  (resolution rate {crr})" if crr is not None else "  (none arose)"))
    L.append(f"  proposal survival         : {rep['proposal_survival_rate']}")
    L.append("")
    L.append("Per unit")
    hdr = (f"  {'unit':<8}{'prop':>5}{'ask':>5}{'recv':>5}{'reply':>6}"
           f"{'obj':>5}{'yield':>6}{'infl':>5}{'infl%':>7}{'conf':>6}{'bad':>5}")
    L.append(hdr)
    for u, d in rep["per_unit"].items():
        L.append(f"  {u:<8}{d['proposals']:>5}{d['requests_sent']:>5}"
                 f"{d['requests_received']:>5}{d['replies_sent']:>6}"
                 f"{d['objections']:>5}{d['yields']:>6}"
                 f"{d['peer_influenced']:>5}{d['peer_influence_rate']:>7.2f}"
                 f"{d['mean_confidence']:>6.2f}{d['parse_failures']:>5}")
    L.append("")
    if rep["message_graph"]:
        L.append("Who talks to whom (directed messages)")
        for src, tgts in rep["message_graph"].items():
            for dst, n in tgts.items():
                L.append(f"  {src} -> {dst}: {n}")
    else:
        L.append("Who talks to whom: NO DIRECTED MESSAGES AT ALL.")
        L.append("  Agents only broadcast. Check use_broadcast / use_negotiation.")
    L.append("")
    L.append("How to read this")
    L.append("  listening rate ~0 with requests>0 => they talk but never adapt;")
    L.append("    ma_comm will not beat ma_nocomm. Strengthen the peer-feed in")
    L.append("    the unit prompt, or raise temperature_unit.")
    L.append("  proposal survival ~1.0 with 0 objections => nobody negotiates;")
    L.append("    conflicts are not being detected. Check detect_conflicts().")
    L.append("=" * width)
    return "\n".join(L)


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="transcript_*.jsonl, or a results directory")
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", action="store_true", help="include raw model output")
    ap.add_argument("--steps", nargs="*", type=int, default=None)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--dialogue-only", action="store_true")
    args = ap.parse_args()

    p = Path(args.path)
    files = (sorted(p.glob("transcript_*.jsonl")) if p.is_dir() else [p])
    if not files:
        print(f"No transcripts under {p}")
        return

    for f in files:
        msgs = load(f)
        if not msgs:
            print(f"\n{f.name}: empty\n")
            continue
        parts = [f"\n### {f.name}\n"]
        if not args.report_only:
            parts.append(render_dialogue(msgs, args.steps, args.raw))
        if not args.dialogue_only:
            parts.append(render_report(collaboration_report(msgs)))
        text = "\n".join(parts)
        if args.out and len(files) == 1:
            Path(args.out).write_text(text, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(text)


if __name__ == "__main__":
    main()