"""Live progress monitor for a RUNNING experiment. Strictly READ-ONLY.

Safe to run while a job is in flight: it only reads results.csv, transcript
JSONL files, and file mtimes. It never touches the simulator directory, never
starts a simulator, and never writes into the results dir.

What it answers
  * how far along is the job, and what is the ETA
  * which (method, episode, trial) are DONE, which one is IN PROGRESS
  * is the in-progress episode still making progress, or has it stalled
  * are the completed episodes healthy (locks, frozen clocks, illegal commands,
    zero switch toggles)
  * for multi-agent arms: are the agents actually talking AND adapting

Why the stall check matters: the runner's step guard terminates a frozen
episode cleanly, but a slow LLM or a wedged simulator can still leave a job
making no forward progress. The newest transcript's mtime and max time step
detect that in seconds.

Usage
  python -m llm_agents.monitor --dir llm_agents/results/multi_v2
  python -m llm_agents.monitor --dir <dir> --methods ma_nocomm ma_comm \
      --episodes 20 --trials 1
  python -m llm_agents.monitor --dir <dir> --watch 60     # refresh every 60s
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

STALL_WARN_S = 300          # no transcript write in 5 min -> warn
STALL_ALERT_S = 900         # no transcript write in 15 min -> alert

SINGLE_AGENT_METHODS = ("noop", "scripted", "scripted_switch", "lara",
                        "gen_only", "gen_ver", "gen_ver_llm")


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


def _hhmm(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def _age(path: Path) -> float:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return float("inf")


# --------------------------------------------------------------------- #
def read_rows(csv_path: Path) -> List[dict]:
    """Tolerant read: the runner may be appending as we read."""
    if not csv_path.exists():
        return []
    try:
        with csv_path.open(newline="", encoding="utf-8") as fh:
            return [r for r in csv.DictReader(fh) if r.get("method")]
    except Exception:
        return []


def transcript_tag(path: Path) -> Tuple[str, str, str]:
    """transcript_<method>_ep<N>_t<M>.jsonl -> (method, ep, trial)."""
    stem = path.stem
    if stem.startswith("transcript_"):
        stem = stem[len("transcript_"):]
    parts = stem.rsplit("_ep", 1)
    if len(parts) != 2:
        return stem, "?", "?"
    method = parts[0]
    rest = parts[1]
    if "_t" in rest:
        ep, trial = rest.split("_t", 1)
    else:
        ep, trial = rest, "0"
    return method, ep, trial


def scan_transcript(path: Path, tail_bytes: int = 400_000) -> Dict[str, Any]:
    """Cheaply summarize a possibly-huge, possibly-partial JSONL transcript.

    We read only the tail for the step/kind summary, so this stays fast even
    on a 100 MB file, and tolerates a half-written final line.
    """
    out = {"messages": 0, "max_t": None, "kinds": Counter(),
            "influenced": 0, "parse_fail": 0, "age_s": _age(path),
            "size_mb": 0.0}
    try:
        size = path.stat().st_size
        out["size_mb"] = size / 2**20
        with path.open("rb") as fh:
            if size > tail_bytes:
                fh.seek(size - tail_bytes)
                fh.readline()        # discard the partial first line
            data = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return out

    for line in data.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue                 # the last line may be mid-write
        out["messages"] += 1
        out["kinds"][str(m.get("kind"))] += 1
        try:
            t = int(m.get("t"))
            if out["max_t"] is None or t > out["max_t"]:
                out["max_t"] = t
        except (TypeError, ValueError):
            pass
        if (m.get("payload") or {}).get("influenced_by"):
            out["influenced"] += 1
        if m.get("parse_ok") is False:
            out["parse_fail"] += 1
    return out


# --------------------------------------------------------------------- #
def report(root: Path, methods: Optional[List[str]], episodes: Optional[int],
           trials: Optional[int], width: int = 96) -> None:
    rows = read_rows(root / "results.csv")
    transcripts = sorted(root.glob("transcript_*.jsonl"))

    done = {(r["method"], str(r.get("episode")), str(r.get("trial") or 0))
            for r in rows}
    by_m: Dict[str, List[dict]] = defaultdict(list)
    for r in rows:
        by_m[r["method"]].append(r)

    print("=" * width)
    print(f"LIVE MONITOR  --  {root}")
    print(f"{time.strftime('%F %T')}   (read-only; safe while the job runs)")
    print("=" * width)

    if not rows and not transcripts:
        print("\nNo results.csv and no transcripts yet. The job is probably")
        print("still loading model weights. Check the slurm .out file.")
        return

    # ---------------- completion ----------------
    observed = sorted(by_m)
    want = methods or observed
    total = (len(want) * (episodes or 0) * (trials or 1)) if episodes else None

    print(f"\nCOMPLETED EPISODES: {len(rows)}"
          + (f" / {total} expected" if total else ""))
    hdr = (f"  {'method':<20}{'done':>6}{'score':>11}{'lock':>6}{'frozen':>7}"
           f"{'illeg':>7}{'sw/ep':>7}{'s/ep':>7}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    wall_by_m: Dict[str, float] = {}
    for m in want:
        rs = by_m.get(m, [])
        if not rs:
            print(f"  {m:<20}{0:>6}{'-':>11}{'-':>6}{'-':>7}{'-':>7}"
                  f"{'-':>7}{'-':>7}")
            continue
        wall = _mean([_f(r, "wall_clock_s") for r in rs])
        wall_by_m[m] = wall
        print(f"  {m:<20}{len(rs):>6}"
              f"{_mean([_f(r,'final_score') for r in rs]):>11,.0f}"
              f"{sum(_b(r,'locked') for r in rs):>6}"
              f"{sum(_b(r,'frozen') for r in rs):>7}"
              f"{_mean([_f(r,'safety_catches') for r in rs]):>7.1f}"
              f"{_mean([_f(r,'switch_toggles') for r in rs]):>7.1f}"
              f"{wall:>7.0f}")

    # ---------------- ETA ----------------
    if total and rows:
        remaining = total - len(rows)
        per_ep = _mean(list(wall_by_m.values())) if wall_by_m else 0.0
        if remaining > 0 and per_ep > 0:
            print(f"\n  remaining: {remaining} episodes  x  ~{per_ep:.0f}s"
                  f"  =  ETA ~{_hhmm(remaining * per_ep)}")

    # ---------------- in progress ----------------
    in_prog = [p for p in transcripts
               if transcript_tag(p)[0:1][0] and
               (transcript_tag(p)[0], transcript_tag(p)[1],
                transcript_tag(p)[2]) not in done]
    print(f"\nIN PROGRESS: {len(in_prog)} transcript(s) with no results row")
    if not in_prog:
        print("  (none -- the job is between episodes, or has finished)")
    for p in sorted(in_prog, key=lambda q: -q.stat().st_mtime)[:4]:
        method, ep, trial = transcript_tag(p)
        s = scan_transcript(p)
        age = s["age_s"]
        state = ("ACTIVE" if age < STALL_WARN_S else
                 "SLOW" if age < STALL_ALERT_S else "STALLED?")
        print(f"  {method} ep{ep} t{trial}:  T={s['max_t']}  "
              f"msgs~{s['messages']}  {s['size_mb']:.1f}MB  "
              f"last write {_hhmm(age)} ago  [{state}]")
        kinds = ", ".join(f"{k}={v}" for k, v in s["kinds"].most_common(6))
        if kinds:
            print(f"      kinds: {kinds}")
        if s["parse_fail"]:
            print(f"      !! {s['parse_fail']} model outputs failed to parse "
                  f"in the tail")
        if state == "STALLED?":
            print(f"      *** no transcript write in {_hhmm(age)}. Check the "
                  f"slurm log; the step guard should have fired by now. ***")

    # ---------------- live health ----------------
    print("\nHEALTH OF COMPLETED EPISODES")
    print("-" * width)
    issues: List[str] = []
    for m, rs in sorted(by_m.items()):
        locked = [r for r in rs if _b(r, "locked")]
        frozen = [r for r in rs if _b(r, "frozen")]
        for r in locked:
            issues.append(f"{m} ep{r.get('episode')} t{r.get('trial')}: LOCKED "
                          f"at T={r.get('lock_step')} losing "
                          f"[{r.get('violation_loads')}]")
        for r in frozen:
            issues.append(f"{m} ep{r.get('episode')} t{r.get('trial')}: FROZEN "
                          f"-- {str(r.get('frozen_reason'))[:90]}")
        if rs and _mean([_f(r, "switch_toggles") for r in rs]) == 0 \
                and m not in ("noop", "scripted"):
            issues.append(f"{m}: ZERO switch toggles across {len(rs)} episodes "
                          f"-- this arm cannot reconnect anything")
        if rs and _mean([_f(r, "safety_catches") for r in rs]) > 2:
            issues.append(f"{m}: "
                          f"{_mean([_f(r,'safety_catches') for r in rs]):.1f} "
                          f"illegal commands/ep")
    print("  no issues" if not issues else "")
    for i in issues[:20]:
        print(f"  * {i}")
    if len(issues) > 20:
        print(f"  ... and {len(issues) - 20} more")

    # ---------------- communication, multi-agent arms only ----------------
    ma = [m for m in by_m if m not in SINGLE_AGENT_METHODS]
    if ma:
        print("\nCOMMUNICATION (multi-agent arms, from completed episodes)")
        chdr = (f"  {'method':<20}{'asks':>7}{'repl':>7}{'obj':>6}{'yield':>7}"
                f"{'infl':>7}{'conf':>12}{'collab':>8}")
        print(chdr)
        print("  " + "-" * (len(chdr) - 2))
        for m in sorted(ma):
            rs = by_m[m]
            cb = _mean([_f(r, "conflicts") for r in rs])
            ca = _mean([_f(r, "conflicts_after") for r in rs])
            collab = [_f(r, "collab_score") for r in rs
                      if r.get("collab_score") not in (None, "")]
            print(f"  {m:<20}{_mean([_f(r,'requests') for r in rs]):>7.1f}"
                  f"{_mean([_f(r,'replies') for r in rs]):>7.1f}"
                  f"{_mean([_f(r,'objections') for r in rs]):>6.1f}"
                  f"{_mean([_f(r,'yields') for r in rs]):>7.1f}"
                  f"{_mean([_f(r,'influenced') for r in rs]):>7.1f}"
                  f"{f'{cb:.1f}->{ca:.1f}':>12}"
                  f"{(f'{_mean(collab):.1f}' if collab else '-'):>8}")
        print("\n  Read `infl` first: proposals that cite a teammate AND change")
        print("  the action. If infl ~0 while obj > 0, the agents are talking")
        print("  without adapting, and ma_comm will not beat ma_nocomm.")

    print("\n" + "=" * width)
    print("next:  python -m llm_agents.analyze --dir " + str(root))
    print("=" * width)


# --------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--methods", nargs="*", default=None,
                    help="expected methods, for completion accounting")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--watch", type=int, default=0,
                    help="refresh every N seconds (0 = print once)")
    args = ap.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"{root} does not exist yet.")
        return
    while True:
        if args.watch:
            os.system("clear" if os.name != "nt" else "cls")
        report(root, args.methods, args.episodes, args.trials)
        if not args.watch:
            break
        time.sleep(max(5, args.watch))


if __name__ == "__main__":
    main()