"""Bundle experiment results into ONE compact, pasteable/attachable report.

Why this exists: results.csv is tiny, but transcript_*.jsonl contains the
verbatim model output for every utterance and can be hundreds of MB. This
script extracts exactly the artifacts needed for diagnosis, strips the `raw`
fields, samples representative episodes, and enforces a character budget.

Usage
  # default: compact report over every results dir found under llm_agents/results
  python -m llm_agents.collect_results --out report.md

  # explicit dirs, bigger budget, more transcripts
  python -m llm_agents.collect_results \
      --dirs llm_agents/results/smoke llm_agents/results/single llm_agents/results/multi \
      --max-chars 250000 --transcripts 4 --out report.md

  # just the numbers, no conversations (tiny)
  python -m llm_agents.collect_results --mode csv-only --out numbers.md

  # everything, no caps (can be very large; for your own archive)
  python -m llm_agents.collect_results --mode full --out report_full.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------- #
# section priorities: lower number = kept first when the budget is tight
# --------------------------------------------------------------------- #
P_INVENTORY = 0
P_CSV = 1
P_AGG = 2
P_DEVICE = 3
P_FAILURES = 4
P_CURVES = 5
P_COMMS = 6
P_REFLECT = 7
P_TRANSCRIPT_GOOD = 8
P_SKILLS = 9
P_TRANSCRIPT_BAD = 10
P_SWITCHLOG = 11

ORDER = {"digest": 0, "proposal": 1, "request": 2, "objection": 3,
         "switchplan": 4, "assignment": 5, "verdict": 6,
         "execution": 7, "reflection": 8, "patch": 9}

NO_LLM = ("noop", "scripted", "scripted_switch")


# --------------------------------------------------------------------- #
# small utils
# --------------------------------------------------------------------- #
def _f(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = row.get(key)
        if v in (None, ""):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _b(row: Dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).strip().lower() in ("true", "1", "yes")


def _mean(xs: List[float]) -> float:
    return st.mean(xs) if xs else 0.0


def _sd(xs: List[float]) -> float:
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def _num(x: float, nd: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x:,.{nd}f}"


def read_rows(path: Path) -> List[dict]:
    if not path.exists():
        return []
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))
    except Exception:
        return []


def read_jsonl(path: Path, limit: Optional[int] = None) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if limit and len(out) >= limit:
                    break
    except Exception:
        pass
    return out


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    if not rows:
        return "_(no rows)_\n"
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return f"{head}\n{sep}\n{body}\n"


# --------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------- #
def discover(roots: List[Path]) -> List[Path]:
    found: List[Path] = []
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        if (root / "results.csv").exists():
            found.append(root)
        for p in sorted(root.rglob("results.csv")):
            d = p.parent
            if d not in found:
                found.append(d)
    return found


# --------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------- #
def sec_inventory(dirs: List[Path]) -> str:
    rows = []
    for d in dirs:
        r = read_rows(d / "results.csv")
        methods = sorted({x.get("method", "?") for x in r})
        models = sorted({x.get("model", "?") for x in r})
        n_tr = len(list(d.glob("transcript_*.jsonl")))
        n_in = len(list(d.glob("interactions_*.jsonl")))
        size_mb = sum(p.stat().st_size for p in d.glob("*") if p.is_file()) / 2**20
        rows.append([str(d), len(r), ", ".join(methods),
                     ", ".join(m.split("/")[-1] for m in models),
                     n_tr, n_in, f"{size_mb:.1f}"])
    return ("## Run inventory\n\n"
            + md_table(["dir", "rows", "methods", "models",
                        "#transcripts", "#interaction logs", "dir size MB"], rows))


def sec_csv(dirs: List[Path], full: bool) -> str:
    out = ["## Raw results.csv (verbatim)\n"]
    for d in dirs:
        p = d / "results.csv"
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8").rstrip()
        n_lines = text.count("\n") + 1
        if not full and n_lines > 400:
            lines = text.splitlines()
            text = "\n".join(lines[:400]) + f"\n... [{n_lines - 400} more rows omitted]"
        out.append(f"### `{p}`\n\n```csv\n{text}\n```\n")
        # also surface any rotated legacy files so stale data is obvious
        for legacy in sorted(d.glob("results_legacy_*.csv")):
            lr = read_rows(legacy)
            out.append(f"_note: rotated legacy file `{legacy.name}` "
                       f"with {len(lr)} rows exists (stale schema, ignored)._\n")
    return "\n".join(out)


def sec_aggregate(dirs: List[Path]) -> str:
    out = ["## Aggregate by (dir, method)\n"]
    headers = ["dir", "method", "n", "score mean", "score sd", "steps",
               "lock%", "fin%", "repaired%", "sw/ep", "mps steps/ep",
               "llm calls/ep", "illegal/ep", "fallback/ep", "execfail/ep",
               "Phi start→end", "s/ep"]
    rows = []
    for d in dirs:
        by_m: Dict[str, List[dict]] = defaultdict(list)
        for r in read_rows(d / "results.csv"):
            by_m[r.get("method", "?")].append(r)
        for m, rs in sorted(by_m.items(),
                            key=lambda kv: -_mean([_f(x, "final_score") for x in kv[1]])):
            sc = [_f(x, "final_score") for x in rs]
            rep = []
            for x in rs:
                nf = _f(x, "n_faults")
                rep.append(_f(x, "faults_repaired") / nf if nf > 0 else 0.0)
            rows.append([
                d.name, m, len(rs), _num(_mean(sc)), _num(_sd(sc)),
                f"{_mean([_f(x,'steps_taken') for x in rs]):.1f}",
                f"{100*_mean([1.0 if _b(x,'locked') else 0.0 for x in rs]):.0f}%",
                f"{100*_mean([1.0 if _b(x,'all_loads_restored') else 0.0 for x in rs]):.0f}%",
                f"{100*_mean(rep):.0f}%",
                f"{_mean([_f(x,'switch_toggles') for x in rs]):.1f}",
                f"{_mean([_f(x,'mps_supplied_steps') for x in rs]):.1f}",
                f"{_mean([_f(x,'llm_calls') for x in rs]):.0f}",
                f"{_mean([_f(x,'safety_catches') for x in rs]):.1f}",
                f"{_mean([_f(x,'fallbacks') for x in rs]):.1f}",
                f"{_mean([_f(x,'exec_failures') for x in rs]):.1f}",
                f"{_mean([_f(x,'phi_start') for x in rs]):.0f}→"
                f"{_mean([_f(x,'phi_end') for x in rs]):.0f}",
                f"{_mean([_f(x,'wall_clock_s') for x in rs]):.0f}",
            ])
    out.append(md_table(headers, rows))

    # paired comparisons on matched scenario_seed (the causal tests)
    pairs = [("ma_nocomm", "ma_comm"), ("ma_comm", "ma_comm_reflect"),
             ("ma_comm_reflect", "ma_full"), ("scripted_switch", "ma_comm"),
             ("scripted_switch", "ma_full"), ("gen_ver", "gen_ver_llm"),
             ("gen_only", "gen_ver"), ("scripted", "scripted_switch")]
    prows = []
    for d in dirs:
        by_m: Dict[str, Dict[Tuple, float]] = defaultdict(dict)
        for r in read_rows(d / "results.csv"):
            key = (r.get("scenario_seed"), r.get("trial"))
            by_m[r.get("method", "?")][key] = _f(r, "final_score")
        for a, b in pairs:
            if a not in by_m or b not in by_m:
                continue
            shared = sorted(set(by_m[a]) & set(by_m[b]))
            if not shared:
                continue
            delta = [by_m[b][k] - by_m[a][k] for k in shared]
            md, sdd = _mean(delta), _sd(delta)
            t = (md / (sdd / math.sqrt(len(delta)))) if sdd > 0 and len(delta) > 1 else float("nan")
            prows.append([d.name, f"{b} − {a}", len(delta), _num(md), _num(sdd),
                          "-" if math.isnan(t) else f"{t:.2f}",
                          f"{sum(1 for x in delta if x > 0)}/{len(delta)}"])
    if prows:
        out.append("\n### Paired comparisons on matched scenario_seed\n")
        out.append(md_table(["dir", "contrast", "n", "mean Δ", "sd", "paired t", "wins"], prows))
    return "\n".join(out)


def sec_device(dirs: List[Path]) -> str:
    """Seconds per LLM call: a reliable CPU-vs-GPU fingerprint.

    GPU 7B ~2-5 s/call. CPU 7B ~60-700 s/call. If a row shows the latter the
    run is unusable for any latency claim.
    """
    rows = []
    for d in dirs:
        by_m: Dict[str, List[dict]] = defaultdict(list)
        for r in read_rows(d / "results.csv"):
            by_m[r.get("method", "?")].append(r)
        for m, rs in sorted(by_m.items()):
            calls = sum(_f(x, "llm_calls") for x in rs)
            wall = sum(_f(x, "wall_clock_s") for x in rs)
            if calls <= 0:
                verdict, spc = "no LLM", "-"
            else:
                v = wall / calls
                spc = f"{v:.1f}"
                verdict = ("plausible GPU" if v < 8 else
                           "SLOW - check device" if v < 30 else
                           "almost certainly CPU")
            rows.append([d.name, m, f"{calls:.0f}", f"{wall:.0f}", spc, verdict])
    return ("## Device sanity (wall-clock seconds per LLM call)\n\n"
            + md_table(["dir", "method", "LLM calls", "total s", "s/call", "verdict"], rows))


def sec_failures(dirs: List[Path], cap: int = 40) -> str:
    out = ["## Failure digest (locks, zeros, fallbacks, exec failures)\n"]
    rows = []
    for d in dirs:
        for r in read_rows(d / "results.csv"):
            bad = []
            if _b(r, "locked"):
                bad.append(f"LOCKED@T{r.get('lock_step')} lost[{r.get('violation_loads')}]")
            if _f(r, "final_score") <= 0 and r.get("method") != "noop":
                bad.append("score=0")
            if _f(r, "fallbacks") > 0:
                bad.append(f"fallbacks={int(_f(r,'fallbacks'))}")
            if _f(r, "exec_failures") > 0:
                bad.append(f"execfail={int(_f(r,'exec_failures'))}")
            if _f(r, "safety_catches") > 2:
                bad.append(f"illegal={int(_f(r,'safety_catches'))}")
            nf = _f(r, "n_faults")
            if nf > 0 and _f(r, "faults_repaired") < nf:
                bad.append(f"repaired {int(_f(r,'faults_repaired'))}/{int(nf)}")
            if _f(r, "unknown_px_end") > 0:
                bad.append(f"unknown_px_end={int(_f(r,'unknown_px_end'))}")
            if bad:
                rows.append([d.name, r.get("method"), r.get("episode"),
                             r.get("trial"), r.get("scenario_seed"),
                             _num(_f(r, "final_score")), "; ".join(bad)])
    if len(rows) > cap:
        out.append(f"_showing {cap} of {len(rows)} flagged episodes_\n")
        rows = rows[:cap]
    out.append(md_table(["dir", "method", "ep", "trial", "seed", "score", "flags"], rows))
    return "\n".join(out)


def sec_curves(dirs: List[Path], per_method: int = 2) -> str:
    """Per-step weighted-load trajectory: shows WHEN score accrues."""
    out = ["## Score trajectories (weighted_after per step)\n",
           "_Reveals whether score comes from early restoration or a late "
           "MPS/grid window. Flat-then-jump means the trajectory barely matters._\n"]
    for d in dirs:
        seen: Counter = Counter()
        for p in sorted(d.glob("trace_*.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:
                continue
            m = (data.get("summary") or {}).get("method", "?")
            if seen[m] >= per_method:
                continue
            seen[m] += 1
            trace = data.get("trace") or []
            w = " ".join(f"T{s.get('t')}:{_num(_f(s,'weighted_after'))}" for s in trace)
            src = Counter(str(s.get("source")) for s in trace)
            out.append(f"**{d.name} / {p.name}** "
                       f"score={(data.get('summary') or {}).get('final_score')} "
                       f"sources={dict(src)}\n\n```\n{w}\n```\n")
    return "\n".join(out)


def sec_comms(dirs: List[Path]) -> str:
    try:
        from llm_agents.metrics import communication_metrics
    except Exception:
        return "## Communication metrics\n\n_(metrics module unavailable)_\n"
    out = ["## Communication metrics (from transcripts)\n"]
    headers = ["dir", "method", "eps", "msgs/step", "llm/step", "mean lat s",
               "schema fail", "proposal survival", "requests", "objections",
               "reflections", "verdict reject"]
    rows = []
    for d in dirs:
        by_m: Dict[str, List[dict]] = defaultdict(list)
        for p in sorted(d.glob("transcript_*.jsonl")):
            stem = p.stem[len("transcript_"):]
            method = stem.split("_ep")[0]
            try:
                c = communication_metrics(p)
            except Exception:
                continue
            if c:
                by_m[method].append(c)
        for m, cs in sorted(by_m.items()):
            g = lambda k: _mean([c[k] for c in cs if isinstance(c.get(k), (int, float))])
            rows.append([d.name, m, len(cs),
                         f"{g('messages_per_step'):.1f}", f"{g('llm_calls_per_step'):.1f}",
                         f"{g('mean_latency_s'):.1f}", f"{g('schema_fail_rate'):.2f}",
                         f"{g('proposal_survival_rate'):.2f}",
                         f"{g('requests_sent'):.0f}", f"{g('objections'):.0f}",
                         f"{g('reflections'):.0f}", f"{g('verdict_reject_rate'):.2f}"])
    out.append(md_table(headers, rows))

    # message-kind census: did anyone actually talk?
    krows = []
    for d in dirs:
        kinds: Dict[str, Counter] = defaultdict(Counter)
        for p in sorted(d.glob("transcript_*.jsonl")):
            method = p.stem[len("transcript_"):].split("_ep")[0]
            for msg in read_jsonl(p):
                kinds[method][str(msg.get("kind"))] += 1
        for m, c in sorted(kinds.items()):
            krows.append([d.name, m, ", ".join(f"{k}={v}" for k, v in c.most_common())])
    if krows:
        out.append("\n### Message-kind census\n")
        out.append(md_table(["dir", "method", "counts"], krows))
    return "\n".join(out)


def render_transcript(path: Path, max_steps: Optional[int] = None,
                      max_chars: Optional[int] = None) -> str:
    msgs = read_jsonl(path)
    if not msgs:
        return f"_(empty transcript {path.name})_\n"
    steps = sorted({m.get("t", 0) for m in msgs})
    if max_steps and len(steps) > max_steps:
        keep = set(steps[:max_steps // 2] + steps[-(max_steps - max_steps // 2):])
        note = (f"_showing {len(keep)} of {len(steps)} steps "
                f"(first/last split)_\n\n")
    else:
        keep, note = set(steps), ""
    lines = [f"**`{path.name}`**\n", note]
    for t in steps:
        if t not in keep:
            continue
        step = [m for m in msgs if m.get("t") == t]
        step.sort(key=lambda m: (m.get("round", 0),
                                 ORDER.get(str(m.get("kind")), 99),
                                 m.get("ts", 0)))
        lines.append(f"**T{t}**")
        for m in step:
            to = m.get("to") or ["ALL"]
            tag = f"{m.get('speaker')} `{m.get('kind')}`"
            if to != ["ALL"]:
                tag += f" -> {','.join(str(x) for x in to)}"
            txt = str(m.get("text") or "").strip().replace("\n", " ")
            if len(txt) > 400:
                txt = txt[:400] + "..."
            lines.append(f"- {tag}" + (f": {txt}" if txt else ""))
            pay = m.get("payload") or {}
            if pay:
                blob = json.dumps(pay, default=str)
                if len(blob) > 700:
                    blob = blob[:700] + "...}"
                lines.append(f"  - `{blob}`")
            if m.get("parse_ok") is False:
                lines.append("  - **PARSE FAILED**")
        lines.append("")
    text = "\n".join(lines)
    if max_chars and len(text) > max_chars:
        text = text[:max_chars] + f"\n_...[truncated at {max_chars} chars]_\n"
    return text


def pick_transcripts(dirs: List[Path], n_per_kind: int) -> Tuple[List[Path], List[Path]]:
    """Return (best, worst) transcripts across all dirs, preferring LLM methods."""
    scored: List[Tuple[float, Path]] = []
    for d in dirs:
        for r in read_rows(d / "results.csv"):
            m = r.get("method", "?")
            if m in NO_LLM:
                continue
            tag = f"{m}_ep{r.get('episode')}_t{r.get('trial')}"
            p = d / f"transcript_{tag}.jsonl"
            if p.exists():
                scored.append((_f(r, "final_score"), p))
    if not scored:
        for d in dirs:
            for p in sorted(d.glob("transcript_*.jsonl"))[:n_per_kind]:
                scored.append((0.0, p))
    scored.sort(key=lambda kv: kv[0])
    worst = [p for _, p in scored[:n_per_kind]]
    best = [p for _, p in scored[::-1][:n_per_kind]]
    return best, worst


def sec_reflections(dirs: List[Path], cap: int = 60) -> str:
    out = ["## Reflexion verbal memory (all lessons)\n"]
    rows = []
    for d in dirs:
        for p in sorted(d.glob("reflections_*.jsonl")):
            for it in read_jsonl(p):
                rows.append([d.name, p.stem.replace("reflections_", ""),
                             ",".join(str(x) for x in (it.get("tags") or [])),
                             str(it.get("lesson", ""))[:300]])
    if not rows:
        out.append("_(none written - reflection only triggers on failure, loop, "
                   "or fallback; zero reflections means the team never failed "
                   "a step, or use_reflection was off)_\n")
        return "\n".join(out)
    if len(rows) > cap:
        out.append(f"_showing {cap} of {len(rows)}_\n")
        rows = rows[:cap]
    out.append(md_table(["dir", "source", "tags", "lesson"], rows))
    return "\n".join(out)


def sec_skills(dirs: List[Path], cap: int = 40) -> str:
    out = ["## Voyager skill library (signatures only)\n"]
    rows = []
    for d in dirs:
        for p in sorted(d.glob("skills_*.jsonl")):
            for it in read_jsonl(p):
                rows.append([d.name, p.stem.replace("skills_", ""),
                             str(it.get("signature", ""))[:90],
                             _num(_f(it, "delta")), _num(_f(it, "potential")),
                             json.dumps(it.get("batch", {}), default=str)[:200]])
    if not rows:
        out.append("_(empty)_\n")
        return "\n".join(out)
    if len(rows) > cap:
        out.append(f"_showing {cap} of {len(rows)}_\n")
        rows = rows[:cap]
    out.append(md_table(["dir", "lib", "signature", "dW", "dPhi", "batch"], rows))
    return "\n".join(out)


def sec_switchlog(dirs: List[Path], per_method: int = 1) -> str:
    out = ["## SwitchOperator logs (sample)\n"]
    for d in dirs:
        seen: Counter = Counter()
        for p in sorted(d.glob("transcript_*.jsonl")):
            method = p.stem[len("transcript_"):].split("_ep")[0]
            if seen[method] >= per_method:
                continue
            msgs = [m for m in read_jsonl(p) if m.get("kind") == "switchplan"]
            if not msgs:
                continue
            seen[method] += 1
            lines = [f"T{m.get('t')}: {str(m.get('text'))[:260]}" for m in msgs[:25]]
            out.append(f"**{d.name} / {p.name}**\n\n```\n" + "\n".join(lines) + "\n```\n")
    return "\n".join(out)


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=["llm_agents/results"])
    ap.add_argument("--out", default="report.md")
    ap.add_argument("--mode", choices=["compact", "full", "csv-only"],
                    default="compact")
    ap.add_argument("--max-chars", type=int, default=160000)
    ap.add_argument("--transcripts", type=int, default=2,
                    help="how many best AND worst transcripts to include")
    ap.add_argument("--transcript-steps", type=int, default=8)
    args = ap.parse_args()

    dirs = discover([Path(x) for x in args.dirs])
    if not dirs:
        print(f"No results.csv found under {args.dirs}")
        return

    full = args.mode == "full"
    budget = 10**9 if full else args.max_chars

    sections: List[Tuple[int, str, str]] = [
        (P_INVENTORY, "inventory", sec_inventory(dirs)),
        (P_CSV, "results.csv", sec_csv(dirs, full)),
    ]
    if args.mode != "csv-only":
        best, worst = pick_transcripts(dirs, args.transcripts)
        tsteps = None if full else args.transcript_steps
        tchars = None if full else 30000
        sections += [
            (P_AGG, "aggregate", sec_aggregate(dirs)),
            (P_DEVICE, "device", sec_device(dirs)),
            (P_FAILURES, "failures", sec_failures(dirs)),
            (P_CURVES, "curves", sec_curves(dirs)),
            (P_COMMS, "comms", sec_comms(dirs)),
            (P_REFLECT, "reflections", sec_reflections(dirs)),
            (P_TRANSCRIPT_GOOD, "best transcripts",
             "## Transcripts: HIGHEST scoring episodes\n\n"
             + "\n".join(render_transcript(p, tsteps, tchars) for p in best)),
            (P_SKILLS, "skills", sec_skills(dirs)),
            (P_TRANSCRIPT_BAD, "worst transcripts",
             "## Transcripts: LOWEST scoring episodes\n\n"
             + "\n".join(render_transcript(p, tsteps, tchars) for p in worst)),
            (P_SWITCHLOG, "switch logs", sec_switchlog(dirs)),
        ]

    header = ("# DSR LLM multi-agent restoration: results bundle\n\n"
              f"- dirs: {', '.join(str(d) for d in dirs)}\n"
              f"- mode: {args.mode}, budget: "
              f"{'none' if full else args.max_chars} chars\n\n")

    kept, dropped, used = [header], [], len(header)
    for _, name, text in sorted(sections, key=lambda s: s[0]):
        if used + len(text) > budget:
            dropped.append(name)
            continue
        kept.append(text)
        used += len(text)

    if dropped:
        kept.append("\n## Omitted for size\n\n"
                    + "".join(f"- {n}\n" for n in dropped)
                    + "\nRe-run with a larger `--max-chars` or `--mode full` "
                      "to include these.\n")

    out = Path(args.out)
    out.write_text("\n".join(kept), encoding="utf-8")
    kb = out.stat().st_size / 1024
    print(f"Wrote {out}  ({kb:.0f} KB, {used:,} chars)")
    if dropped:
        print(f"Omitted sections: {', '.join(dropped)}")
    if kb > 400:
        print("WARNING: >400 KB may be awkward to paste. Attach the file "
              "instead, or lower --max-chars / --transcripts.")


if __name__ == "__main__":
    main()