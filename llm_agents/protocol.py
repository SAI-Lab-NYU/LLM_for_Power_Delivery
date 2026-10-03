"""Wire format for inter-agent communication + deterministic conflict detection.

Every utterance in the system is a Message. The transcript is the ordered list of
Messages for one time step. This is both the coordination substrate and the
evaluation artifact (see metrics.py / transcript.py).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------- roles
ROLE_DISPATCHER = "Dispatcher"
ROLE_UNIT = "Unit"
ROLE_SAFETY = "SafetyOfficer"
ROLE_REFLECTOR = "Reflector"
ROLE_ENV = "Environment"
ROLE_SWITCH = "SwitchOperator"
ROLE_HUMAN = "Operator"

# ---------------------------------------------------------------- kinds
K_DIGEST = "digest"          # Dispatcher -> ALL : compressed situation + intents
K_PROPOSAL = "proposal"      # Unit -> ALL       : thought + one command
K_REQUEST = "request"        # Unit -> Unit      : ask a peer to do / not do something
K_OBJECTION = "objection"    # Unit -> Unit      : conflict push-back
K_ASSIGNMENT = "assignment"  # Dispatcher -> ALL : the assembled joint batch
K_SWITCHPLAN = "switchplan"  # SwitchOperator    : validated switch toggles
K_VERDICT = "verdict"        # SafetyOfficer     : accept / revise / reject
K_REFLECTION = "reflection"  # Reflector         : verbal lesson for memory
K_EXECUTION = "execution"    # Environment       : what actually happened
K_PATCH = "patch"            # Operator          : mid-episode human update


@dataclass
class Message:
    t: int                                  # simulator time step
    round: int                              # dialogue round within the step
    speaker: str                            # "RC1", "Dispatcher", ...
    role: str
    kind: str
    to: List[str] = field(default_factory=lambda: ["ALL"])
    text: str = ""                          # natural-language content (the "thought")
    payload: Dict[str, Any] = field(default_factory=dict)
    # instrumentation
    prompt_chars: int = 0
    response_chars: int = 0
    latency_s: float = 0.0
    parse_ok: bool = True
    raw: str = ""                           # verbatim model output (truncated on write)
    ts: float = field(default_factory=time.time)

    def visible_to(self, agent: str) -> bool:
        return "ALL" in self.to or agent in self.to or agent == self.speaker

    def to_json(self, raw_limit: int = 4000) -> str:
        d = asdict(self)
        if isinstance(d.get("raw"), str) and len(d["raw"]) > raw_limit:
            d["raw"] = d["raw"][:raw_limit] + f"...[{len(d['raw'])} chars]"
        return json.dumps(d, default=str)

    def render(self) -> str:
        head = f"[T{self.t} R{self.round}] {self.speaker} ({self.kind})"
        if self.to != ["ALL"]:
            head += f" -> {','.join(self.to)}"
        body = self.text.strip()
        pay = json.dumps(self.payload, sort_keys=True) if self.payload else ""
        out = head
        if body:
            out += f"\n    {body}"
        if pay and pay != "{}":
            out += f"\n    >> {pay}"
        return out


# ---------------------------------------------------------------- conflicts
def target_key(command: Dict[str, Any]) -> Optional[Tuple[str, Any]]:
    """Canonical 'what this unit is going after', for conflict detection."""
    if not isinstance(command, dict):
        return None
    ctype = str(command.get("type", "")).lower()
    if ctype in ("repair", "repaire"):
        return ("fault", command.get("fault"))
    if ctype == "move":
        wps = command.get("waypoints") or []
        if wps:
            return ("goto", tuple(int(v) for v in wps[-1]))
    if ctype == "supply":
        return ("supply", "island")
    return None


def detect_conflicts(proposals: Dict[str, Dict[str, Any]],
                     state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return a list of conflicts: duplicated targets, and illegal disturbances.

    Deterministic, cheap, and run BEFORE any extra LLM turn. Only agents named in
    a conflict are invited to speak again (DyLAN-style agent selection).
    """
    objects = state.get("objects", {}) or {}
    by_target: Dict[Tuple[str, Any], List[str]] = {}
    conflicts: List[Dict[str, Any]] = []

    for name, cmd in proposals.items():
        unit = objects.get(name, {}) or {}
        ctype = str((cmd or {}).get("type", "")).lower()
        state_l = str(unit.get("state", "")).strip().lower()

        # hard safety catches (these are bugs, not negotiations)
        if unit.get("analysis_locked") and ctype not in ("", "continue", "keep"):
            conflicts.append({"kind": "analysis_lock_violation", "units": [name],
                              "detail": f"{name} is analysis-locked; only continue is legal."})
        if unit.get("type") == "MPS" and state_l == "supplying" and ctype not in ("", "continue", "keep"):
            conflicts.append({"kind": "supplying_mps_disturbed", "units": [name],
                              "detail": f"{name} is Supplying; {ctype} disconnects it and "
                                        f"de-energizes its island -> permanent score lock."})

        key = target_key(cmd or {})
        if key is not None:
            by_target.setdefault(key, []).append(name)

    for key, units in by_target.items():
        if len(units) > 1 and key[0] in ("fault", "goto"):
            conflicts.append({"kind": "duplicate_target", "units": sorted(units),
                              "detail": f"{', '.join(sorted(units))} all target {key[1]}."})
    return conflicts


def conflicted_units(conflicts: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for c in conflicts:
        for u in c.get("units", []):
            if u not in out:
                out.append(u)
    return out