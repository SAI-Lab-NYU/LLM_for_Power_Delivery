"""Three-tier memory.

  trajectory     : short-term, this episode (Reflexion's short-term memory)
  reflections    : verbal lessons across trials, capped at OMEGA (Reflexion)
  skill library  : validated macro-plans keyed by a situation signature (Voyager)
"""
from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple


# ------------------------------------------------------------------ trajectory
class TrajectoryMemory:
    def __init__(self, keep: int = 4):
        self.keep = keep
        self.items: Deque[str] = deque(maxlen=keep)

    def add(self, t: int, batch: Dict[str, Any], delta: float, note: str = ""):
        mob = (batch or {}).get("mobile_commands") or {}
        sw = (batch or {}).get("switch_commands") or {}
        compact = "; ".join(
            f"{k}:{v.get('type')}" + (f"({v.get('fault')})" if v.get("fault") else "")
            for k, v in mob.items() if isinstance(v, dict))
        self.items.append(f"T{t}: switches={sw or '{}'} | {compact} | "
                          f"d|P|W={delta:+g}{(' | ' + note) if note else ''}")

    def render(self) -> str:
        return "\n".join(self.items) if self.items else "(first step)"


# ------------------------------------------------------------------ reflections
class ReflectionMemory:
    """Reflexion: amplify a sparse outcome signal into natural-language lessons."""

    OMEGA = 3

    def __init__(self, path: Optional[Path] = None, omega: int = OMEGA):
        self.path = Path(path) if path else None
        self.omega = omega
        self.items: List[Dict[str, Any]] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def add(self, lesson: str, tags: Optional[List[str]] = None, meta: Optional[dict] = None):
        lesson = (lesson or "").strip()
        if not lesson:
            return
        item = {"lesson": lesson, "tags": tags or [], "meta": meta or {}}
        self.items.append(item)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item) + "\n")

    def render(self) -> str:
        if not self.items:
            return "(no prior lessons)"
        recent = self.items[-self.omega:]
        return "\n".join(f"- {it['lesson']}" for it in recent)


# ------------------------------------------------------------------ skills
def situation_signature(state: Dict[str, Any], affordances: Dict[str, Any]) -> str:
    """Coarse, discrete key. Deliberately lossy so that it generalizes."""
    game = state.get("game") or {}
    summary = state.get("summary") or {}
    faults = affordances.get("known_active_faults") or []
    objects = state.get("objects", {}) or {}
    n_analyzed = sum(1 for f in faults if f.get("analyzed"))
    n_locked = sum(1 for u in objects.values() if u.get("analysis_locked"))
    mps = ",".join(sorted(str(u.get("state", "")).lower()
                          for u in objects.values() if u.get("type") == "MPS"))
    phase = "pre_grid" if str(summary.get("source_status", "")).lower() != "available" else "post_grid"
    return (f"{phase}|faults={len(faults)}/{n_analyzed}|locked={n_locked}"
            f"|unserved={len(affordances.get('unenergized_loads') or [])}"
            f"|mps={mps or 'none'}|sw={len(affordances.get('switch_toggle_options') or {})}")


def _tokens(s: str) -> set:
    return set(re.findall(r"[a-z0-9_]+", s.lower()))


class SkillLibrary:
    """Voyager-style: only store plans the simulator VALIDATED and that produced
    non-negative progress with zero continuity violations."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path else None
        self.items: List[Dict[str, Any]] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    self.items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    def add(self, signature: str, description: str, batch: Dict[str, Any],
            delta: float, potential: float):
        item = {"signature": signature, "description": description,
                "batch": batch, "delta": delta, "potential": potential}
        # keep only the best plan per exact signature
        for i, existing in enumerate(self.items):
            if existing["signature"] == signature:
                if (delta, potential) <= (existing["delta"], existing["potential"]):
                    return
                self.items[i] = item
                break
        else:
            self.items.append(item)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item) + "\n")

    def retrieve(self, signature: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.items:
            return []
        q = _tokens(signature)
        scored = []
        for it in self.items:
            s = _tokens(it["signature"])
            jac = len(q & s) / max(len(q | s), 1)
            scored.append((jac + 1e-6 * it["delta"], it))
        scored.sort(key=lambda kv: -kv[0])
        return [it for _, it in scored[:k]]

    def render(self, signature: str, k: int = 3) -> str:
        hits = self.retrieve(signature, k)
        if not hits:
            return "(no comparable past plan)"
        return "\n".join(
            f"- in `{h['signature']}` -> {json.dumps(h['batch'])}  (d|P|W={h['delta']:+g})"
            for h in hits)


# ------------------------------------------------------------------ loop detector
class LoopDetector:
    """Reflexion's two dominant failure modes: hallucinated state and inefficient
    planning. Their heuristic: same action + same response for >3 cycles."""

    def __init__(self, cycles: int = 3):
        self.cycles = cycles
        self.history: List[Tuple[str, str]] = []

    def observe(self, batch_key: str, response_key: str) -> Optional[str]:
        self.history.append((batch_key, response_key))
        if len(self.history) < self.cycles:
            return None
        tail = self.history[-self.cycles:]
        if len(set(tail)) == 1:
            return (f"inefficient planning: the same command batch produced the same "
                    f"outcome {self.cycles} steps in a row ({batch_key}).")
        return None