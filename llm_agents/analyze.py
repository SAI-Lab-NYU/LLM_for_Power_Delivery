"""Aggregate results.csv + transcripts into the tables for the paper.

CHANGES IN THIS VERSION
  1. COLLABORATION METRICS ARE N/A FOR SINGLE-AGENT ARMS. `lara` emits one
     `proposal` message per decision point, so the old `n_prop > 0` gate let it
     through and it printed collaboration_score = 25.0. That number is an
     artefact: the conflict-resolution term defaults to 1.0 when no conflicts
     arise, and LARA is architecturally a SINGLE planner emitting one joint
     macro-action [1] -- there is no inter-agent dialogue to score. Single-agent
     arms are now listed by name and print '-'.
  2. PAIRS includes the contrasts that matter and were silently missing, in
     particular (scripted_switch, lara) -- the LARA-vs-baseline paired test,
     which is the headline number and never printed.
  3. A `sig` column reports whether |t| exceeds the two-sided 0.05 critical
     value, so "103% of greedy" is never mistaken for a significant result.
  4. %ref is labelled with its reference method. LARA's published 95.58% on
     IEEE-13 is against time-relaxed MILP, which reaches optimality well within
     the cap on that feeder and so acts as a genuine oracle [1]; a greedy
     heuristic reference is weaker and must be reported as such.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from llm_agents.conversation import collaboration_report, load as load_msgs
from llm_agents.metrics import communication_metrics

BASELINE_PREFERENCE = ["scripted_switch", "scripted", "noop"]

# Arms with no inter-agent dialogue. Every collaboration metric is undefined
# for these; printing a number invites a false comparison against the ma_* arms.
SINGLE_AGENT_METHODS = ("noop", "scripted", "scripted_switch", "lara",
                        "gen_only", "gen_ver", "gen_ver_llm")

MULTI_AGENT_METHODS = ("ma_nocomm", "ma_comm", "ma_comm_reflect", "ma_full")

PAIRS = [
    # environment sanity
    ("scripted", "scripted_switch"),
    # single-agent ladder
    ("gen_only", "gen_ver"),
    ("gen_ver", "gen_ver_llm"),
    # LARA reproduction vs the environment baseline  <-- headline
    ("scripted_switch", "lara"),
    # multi-agent ladder
    ("ma_nocomm", "ma_comm"),            # <-- the communication causal test
    ("ma_comm", "ma_comm_reflect"),
    ("ma_comm_reflect", "ma_full"),
    # multi-agent vs single-agent and vs baseline
    ("lara", "ma_comm"),
    ("lara", "ma_full"),
    ("scripted_switch", "ma_comm"),
    ("scripted_switch", "ma_full"),
]

# two-sided 0.05 critical t by degrees of freedom
T_CRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36,
          8: 2.31, 9: 2.26, 10: 2.23, 11: 2.20, 12: 2.18, 13: 2.16, 14: 2.14,
          15: 2.13, 16: 2.12, 17: 2.11, 18: 2.10, 19: 2.09, 20: 2.09,
          24: 2.06, 29: 2.05}


# --------------------------------------------------------------------- #
def _f(row, key, default=0.0) -> float:
    try:
        v = row.get(key)
        return default if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return default


def _b(row, key) -> bool:
    return str(row.get(key, "")).strip().lower() in ("true", "1", "yes")


def _mean(xs: List[float]) -> float:
    return st.mean(xs) if xs else 0.0


def _sd(xs: List[float]) -> float:
    return st.pstdev(xs) if len(xs) > 1 else 0.0


def _tcrit(n: int) -> float:
    df = max(1, n - 1)
    if df in T_CRIT:
        return T_CRIT[df]
    keys = sorted(T_CRIT)
    for k in keys:
        if df <= k:
            return T_CRIT[k]
    return 1.96


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--no-collab", action="store_true",
                    help="skip the collaboration table (faster on big dirs)")
    args = ap.parse_args()

    root = Path(args.dir)
    csv_path = root / args.csv
    if not csv_path.exists():
        print(f"No {csv_path}.")
        return
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print(f"{csv_path} is empty.")
        return

    by_m: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_m[r.get("method", "?")].append(r)
    means = {m: _mean([_f(r, "final_score") for r in rs])
             for m, rs in by_m.items()}

    base_name = args.baseline
    if base_name is None:
        for cand in BASELINE_PREFERENCE:
            if cand in by_m:
                base_name = cand
                break
    base = means.get(base_name or "", 0.0)
    if base <= 0 and means:
        alt = max(means.items(), key=lambda kv: kv[1])
        if alt[1] > 0:
            print(f"[warn] baseline `{base_name}` mean {base:g}; "
                  f"using `{alt[0]}`.")
            base_name, base = alt

    print(f"\n%ref is relative to `{base_name}` (mean {base:,.0f}).")
    print("NOTE: this is a greedy-heuristic reference, NOT a MILP oracle; it "
          "is not comparable to published %MILP figures.\n")

    # ---------------- task table ----------------
    hdr = (f"{'method':<20}{'score':>10}{'±sd':>9}{'%ref':>8}{'n':>4}"
           f"{'steps':>7}{'lock%':>7}{'fin%':>6}{'rep%':>6}{'undisc':>8}"
           f"{'sw/ep':>7}{'s/ep':>7}{'LLM/st':>8}{'illeg':>7}{'frozen':>8}")
    print(hdr)
    print("-" * len(hdr))
    summary: Dict[str, Any] = {}
    for m, rs in sorted(by_m.items(), key=lambda kv: -means.get(kv[0], 0)):
        sc = [_f(r, "final_score") for r in rs]
        rep = []
        for r in rs:
            nf = _f(r, "n_faults")
            rep.append(_f(r, "faults_repaired") / nf if nf > 0 else 0.0)
        steps = [_f(r, "steps_taken", 1.0) or 1.0 for r in rs]
        pct = (100.0 * _mean(sc) / base) if base > 0 else float("nan")
        frozen = _mean([1.0 if _b(r, "frozen") else 0.0 for r in rs])
        print(f"{m:<20}{_mean(sc):>10,.0f}{_sd(sc):>9,.0f}"
              f"{(f'{pct:6.1f}%' if not math.isnan(pct) else '     -'):>8}"
              f"{len(rs):>4}{_mean(steps):>7.1f}"
              f"{100*_mean([1.0 if _b(r,'locked') else 0.0 for r in rs]):>6.0f}%"
              f"{100*_mean([1.0 if _b(r,'all_loads_restored') else 0.0 for r in rs]):>5.0f}%"
              f"{100*_mean(rep):>5.0f}%"
              f"{_mean([_f(r,'faults_undiscovered') for r in rs]):>8.1f}"
              f"{_mean([_f(r,'switch_toggles') for r in rs]):>7.1f}"
              f"{_mean([_f(r,'wall_clock_s') for r in rs]):>7.0f}"
              f"{_mean([_f(r,'llm_calls')/(_f(r,'steps_taken',1.0) or 1.0) for r in rs]):>8.1f}"
              f"{_mean([_f(r,'safety_catches') for r in rs]):>7.1f}"
              f"{100*frozen:>7.0f}%")
        summary[m] = {
            "score_mean": _mean(sc), "score_sd": _sd(sc), "n": len(rs),
            "pct_ref": None if math.isnan(pct) else round(pct, 2),
            "lock_rate": _mean([1.0 if _b(r, "locked") else 0.0 for r in rs]),
            "frozen_rate": frozen,
            "repair_rate": _mean(rep),
            "switch_toggles": _mean([_f(r, "switch_toggles") for r in rs]),
            "illegal": _mean([_f(r, "safety_catches") for r in rs]),
            "decision_points": _mean([_f(r, "decision_points") for r in rs]),
            "median_latency_s": _mean([_f(r, "median_decision_latency_s")
                                       for r in rs]),
            "single_agent": m in SINGLE_AGENT_METHODS,
        }

    # ---------------- collaboration table ----------------
    if not args.no_collab:
        print("\nCOLLABORATION")
        print("  '-' = NOT APPLICABLE. These arms are single planners with no")
        print("  inter-agent dialogue, so every collaboration metric is")
        print("  undefined: " + ", ".join(SINGLE_AGENT_METHODS))
        chdr = (f"{'method':<20}{'collab':>8}{'msg/st':>8}{'asks':>6}{'repl':>6}"
                f"{'obj':>5}{'yield':>6}{'listen':>8}{'recip':>7}"
                f"{'conf':>12}{'surv':>7}{'schema':>8}")
        print(chdr)
        print("-" * len(chdr))
        for m in sorted(by_m):
            reps, comms = [], []
            for p in sorted(root.glob(f"transcript_{m}_ep*.jsonl")):
                try:
                    msgs = load_msgs(p)
                except Exception:
                    continue
                if msgs:
                    try:
                        reps.append(collaboration_report(msgs))
                    except Exception:
                        pass
                try:
                    c = communication_metrics(p)
                    if c:
                        comms.append(c)
                except Exception:
                    pass
            if not reps and not comms:
                continue

            g = lambda k: _mean([float(r[k]) for r in reps
                                 if isinstance(r.get(k), (int, float))])
            gc = lambda k: _mean([float(c[k]) for c in comms
                                  if isinstance(c.get(k), (int, float))])
            n_prop = g("proposals") if reps else 0.0
            schema = gc("schema_fail_rate")

            # SINGLE-AGENT ARMS: dashes, not a fabricated score.
            if m in SINGLE_AGENT_METHODS or n_prop <= 0:
                print(f"{m:<20}{'-':>8}{g('messages_per_step'):>8.1f}"
                      f"{'-':>6}{'-':>6}{'-':>5}{'-':>6}{'-':>8}{'-':>7}"
                      f"{'-':>12}{'-':>7}{schema:>8.2f}")
                summary.setdefault(m, {}).update({
                    "collaboration_score": None,
                    "collaboration_na_reason": "single planner; no dialogue",
                    "schema_fail_rate": schema})
                continue

            cb, ca = g("conflicts_detected"), g("conflicts_remaining")
            print(f"{m:<20}{g('collaboration_score'):>8.1f}"
                  f"{g('messages_per_step'):>8.1f}{g('requests'):>6.0f}"
                  f"{g('replies'):>6.0f}{g('objections'):>5.0f}"
                  f"{g('yields'):>6.0f}"
                  f"{g('listening_rate'):>8.2f}{g('reciprocity_rate'):>7.2f}"
                  f"{f'{cb:.0f}->{ca:.0f}':>12}"
                  f"{g('proposal_survival_rate'):>7.2f}{schema:>8.2f}")
            summary.setdefault(m, {}).update({
                "collaboration_score": g("collaboration_score"),
                "listening_rate": g("listening_rate"),
                "reciprocity_rate": g("reciprocity_rate"),
                "objections": g("objections"), "yields": g("yields"),
                "requests": g("requests"), "replies": g("replies"),
                "proposals": n_prop,
                "conflicts_detected": cb, "conflicts_remaining": ca,
                "schema_fail_rate": schema,
                "verdict": reps[0].get("collaboration_verdict")})

    # ---------------- diagnostics ----------------
    print("\nDIAGNOSTICS")
    print("-" * 96)
    for m, rs in sorted(by_m.items()):
        locked = [r for r in rs if _b(r, "locked")]
        frozen = [r for r in rs if _b(r, "frozen")]
        extra = ""
        if summary.get(m, {}).get("decision_points"):
            extra = (f"  dp/ep={summary[m]['decision_points']:.1f}"
                     f"  lat={summary[m]['median_latency_s']:.1f}s")
        print(f"{m:<20} locked {len(locked)}/{len(rs)}  "
              f"frozen {len(frozen)}/{len(rs)}"
              f"  sw/ep={_mean([_f(r,'switch_toggles') for r in rs]):.1f}"
              f"  mps_steps/ep={_mean([_f(r,'mps_supplied_steps') for r in rs]):.1f}"
              f"  execfail/ep={_mean([_f(r,'exec_failures') for r in rs]):.1f}"
              f"  fallback/ep={_mean([_f(r,'fallbacks') for r in rs]):.1f}"
              f"  Phi {_mean([_f(r,'phi_start') for r in rs]):.0f}"
              f"->{_mean([_f(r,'phi_end') for r in rs]):.0f}{extra}")
        for r in locked[:3]:
            print(f"    LOCK ep{r.get('episode')} t{r.get('trial')} "
                  f"at T={r.get('lock_step')} losing [{r.get('violation_loads')}]")
        for r in frozen[:3]:
            print(f"    FROZEN ep{r.get('episode')} t{r.get('trial')}: "
                  f"{str(r.get('frozen_reason'))[:110]}")

    # ---------------- red flags ----------------
    print("\nRED FLAGS")
    print("-" * 96)
    flags: List[str] = []
    for m, s in summary.items():
        if m.startswith("_"):
            continue
        if s.get("lock_rate", 0) > 0:
            flags.append(f"{m}: locks in {100*s['lock_rate']:.0f}% of episodes.")
        if s.get("frozen_rate", 0) > 0:
            flags.append(f"{m}: {100*s['frozen_rate']:.0f}% of episodes aborted "
                         f"with a frozen clock.")
        if (m.startswith(("gen_", "ma_")) or m == "lara") \
                and s.get("switch_toggles", 1) == 0:
            flags.append(f"{m}: ZERO switch toggles -- this arm cannot "
                         f"reconnect anything and is capped at the MPS island.")
        if s.get("illegal", 0) > 2:
            flags.append(f"{m}: {s['illegal']:.1f} illegal commands/ep -- the "
                         f"model is ignoring the affordance menus.")
        if s.get("repair_rate", 1) < 0.34 and m != "noop":
            flags.append(f"{m}: only {100*s['repair_rate']:.0f}% of faults repaired.")
        if s.get("schema_fail_rate") is not None and s["schema_fail_rate"] > 0.25:
            flags.append(f"{m}: schema failure rate "
                         f"{s['schema_fail_rate']:.2f} -- the prompt is probably "
                         f"too long for this backbone.")
        # collaboration flags apply ONLY to multi-agent arms
        if m in MULTI_AGENT_METHODS:
            ls, obj = s.get("listening_rate"), s.get("objections")
            if ls is not None and obj is not None and obj > 0 and ls < 0.1:
                flags.append(f"{m}: agents talk ({obj:.0f} objections) but "
                             f"listening rate is {ls:.2f} -- TALKING WITHOUT "
                             f"ADAPTING; ma_comm will not beat ma_nocomm.")
            cs = s.get("collaboration_score")
            if cs is not None and cs < 20 and not m.endswith("nocomm"):
                flags.append(f"{m}: collaboration score {cs:.0f}/100 -- "
                             f"parallel monologue, not teamwork.")
        if (s.get("n", 0) > 2 and s.get("score_mean", 0) > 0
                and s["score_sd"] / max(s["score_mean"], 1) < 0.01):
            flags.append(f"{m}: score identical across seeds (sd/mean<1%); the "
                         f"policy is scenario-insensitive.")
    print("  none" if not flags else "")
    for f in flags:
        print(f"  * {f}")

    # ---------------- paired tests ----------------
    printed = False
    for a, b in PAIRS:
        if a not in by_m or b not in by_m:
            continue
        ka = {(r.get("scenario_seed"), r.get("trial")): _f(r, "final_score")
              for r in by_m[a]}
        kb = {(r.get("scenario_seed"), r.get("trial")): _f(r, "final_score")
              for r in by_m[b]}
        shared = sorted(set(ka) & set(kb))
        if not shared:
            continue
        if not printed:
            print("\nPAIRED COMPARISONS (matched scenario_seed, trial)")
            print(f"  {'contrast':<36}{'n':>4}{'mean Δ':>12}{'sd':>10}"
                  f"{'t':>8}{'wins':>9}{'sig':>5}")
            printed = True
        d = [kb[k] - ka[k] for k in shared]
        md, sdd = _mean(d), _sd(d)
        tst = (md / (sdd / math.sqrt(len(d)))) if sdd > 0 and len(d) > 1 \
            else float("nan")
        crit = _tcrit(len(d))
        sig = "-" if math.isnan(tst) else ("yes" if abs(tst) > crit else "no")
        wins = sum(1 for x in d if x > 0)
        ties = sum(1 for x in d if x == 0)
        print(f"  {b + ' − ' + a:<36}{len(d):>4}{md:>12,.0f}{sdd:>10,.0f}"
              f"{('-' if math.isnan(tst) else f'{tst:.2f}'):>8}"
              f"{f'{wins}/{len(d)-ties}':>9}{sig:>5}")
        summary[f"_paired_{b}_vs_{a}"] = {
            "n": len(d), "mean_delta": md, "sd": sdd,
            "t": None if math.isnan(tst) else tst,
            "t_crit_p05": crit,
            "significant_p05": None if math.isnan(tst) else bool(abs(tst) > crit),
            "wins": wins, "ties": ties}

    (root / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {root/'summary.json'}")


if __name__ == "__main__":
    main()