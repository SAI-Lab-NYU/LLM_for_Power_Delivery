"""Faithful LARA baseline (Findings of the ACL: EACL 2026).

LARA = affordance-first, language-conditioned planning over a compact,
PRE-VALIDATED catalogue of feasible actions, with greedy switch
reconfiguration kept inside the ENVIRONMENT rather than in the LLM [1].

Why a separate module instead of reusing our `gen_ver_llm` arm: three
differences matter for faithfulness, and the third is the one that produced a
wrong latency metric in our first attempt.

  1. PLAN SAMPLING. k diverse plans are sampled IN PARALLEL and then
     DEDUPLICATED before scoring (paper Sec. 2.5). The paper reports k=3 and
     T=0.2 as optimal, with a sharp drop at T=0.8, and notes that k=5 gives
     little further gain because the LLM returns fairly similar plans at low
     temperature [1].

  2. SELECTION BY ROLLOUT. Each candidate is evaluated by a short rollout
     under the same physical model used in precomputation, together with the
     greedy switching module; the rollout runs until the first reassignment
     trigger, a short-horizon restoration score consistent with the objective
     is computed, and the highest scorer wins, with ties resolved in favour of
     EARLIER proposals (paper Sec. 2.6) [1].

  3. EXECUTE-UNTIL-IDLE. The chosen plan is executed step by step until an
     agent becomes idle, at which point a reassignment request is issued to
     the LLM (paper Sec. 2.6) [1]. Our first implementation tested
     `state == "Idle"`, which is permanently true for an MPS holding position
     or a crew out of resources, so it replanned on EVERY step: 13 decision
     points in a 13-step episode, ~3x the LLM calls, and a decision-latency
     figure that is not the paper's metric. `needs_reassignment` now fires
     only on a TRANSITION into idle or on the completion of a planned action.

DOCUMENTED DEVIATION. The paper rolls forward under the same physical model
until the first reassignment trigger [1]. The V23 simulator exposes
/api/text/validate (exactly one step, on a deep copy) but no forkable
multi-step rollout, so `_rollout` uses the exact one-step prediction plus the
dense progress potential Phi as the continuation proxy. The one-step part is
exact: it already enforces radiality, unknown-area safety, source-restoration
safety and the per-load continuity rule. The continuation part is a proxy.
A true multi-step rollout would require adding a /api/text/rollout endpoint to
the simulator, which is outside the llm_agents/ boundary.

METRICS. We report `decision_points` and the MEDIAN wall-clock time at a
single decision point, matching the paper's definition of decision latency,
which must stay below the restoration timestep duration to avoid delaying
execution in real deployments [1].
"""
from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llm_agents.affordances import (build_affordances, recommend,
                                    recommended_command, situation_card)
from llm_agents.client import LLMClient, extract_json
from llm_agents.metrics import potential
from llm_agents.switching import SwitchOperator

NOOP: Dict[str, Any] = {"switch_commands": {}, "mobile_commands": {}}

# Weight on the dense progress potential relative to realised weighted load.
# Kept identical to switching.py so the two selection signals agree.
LAMBDA_PHI = 0.25

# Safety valve: force a replan if a committed plan has run this many steps
# without any reassignment trigger. Guards against a plan silently going stale
# (e.g. a crew walking to a fault that another unit already repaired).
MAX_PLAN_AGE = 8


LARA_SYSTEM = """You are the restoration planner for a post-disaster power
distribution network. You output ONE high-level plan over a PRE-VALIDATED
catalogue of feasible actions. Every ID you emit must appear verbatim in the
catalogue; the simulator validates all IDs before execution.

Objective: maximize the sum over time of sum(|P| * W) over ENERGIZED loads.
Restoring high-weight load EARLIER is worth more: it re-counts at every later
step. FATAL: if a load that was energized ever loses power, the score is
permanently locked to zero. Prefer safe, monotone restoration.

Switch reconfiguration is handled by the environment with a fast greedy
procedure. Do NOT plan switch closures. Leave "switch_commands" empty.

Output format EXACTLY, and nothing else:
{"reasoning":"<=3 sentences on the coordination strategy",
 "switch_commands": {},
 "mobile_commands": {"RC1": {"type":"continue"}}}

Rules:
- Choose ONLY from the provided legal command menu / move waypoints.
- Omitted units default to "continue".
- An analysis-locked unit accepts ONLY {"type":"continue"}.
- A Supplying MPS accepts ONLY {"type":"continue"}.
- MPS supply on an islanded cluster must EXACTLY match the cluster P and Q, and
  the MPS must be able to hold that output until the final step, or the score
  locks to zero.
- A RECOMMENDED ACTION is given per unit. Follow it unless you state a concrete
  reason to deviate.

Return JSON only. No prose outside the JSON."""


# --------------------------------------------------------------------- #
@dataclass
class LARAPlan:
    command: Dict[str, Any] = field(default_factory=lambda: dict(NOOP))
    reasoning: str = ""
    index: int = 0                  # proposal order, for tie-breaking
    parse_ok: bool = True
    raw: str = ""
    latency_s: float = 0.0
    # rollout outcome
    rollout_score: float = float("-inf")
    delta_weighted: float = 0.0
    delta_potential: float = 0.0
    rollout_steps: int = 0
    fatal: bool = False
    sim_ok: bool = False
    sim_message: str = ""
    switch_report: List[str] = field(default_factory=list)


# --------------------------------------------------------------------- #
def _canon(cmd: Dict[str, Any]) -> str:
    """Canonical string form of a plan, for deduplication (paper Sec. 2.5)."""
    mob = (cmd or {}).get("mobile_commands") or {}
    parts: List[str] = []
    for name in sorted(mob):
        c = mob[name]
        if not isinstance(c, dict):
            continue
        t = str(c.get("type", "")).strip().lower()
        if t == "repaire":
            t = "repair"
        if t == "move":
            wps: List[Tuple[int, int]] = []
            for p in (c.get("waypoints") or []):
                try:
                    wps.append((int(p[0]), int(p[1])))
                except Exception:
                    continue
            parts.append(f"{name}:move{tuple(wps)}")
        elif t == "repair":
            parts.append(f"{name}:repair:{c.get('fault')}:{c.get('amount')}")
        elif t == "supply":
            parts.append(f"{name}:supply:{c.get('p')}:{c.get('q')}")
        else:
            parts.append(f"{name}:{t or 'continue'}")
    return "|".join(parts)


def _normalize(obj: Any) -> Dict[str, Any]:
    """Coerce a parsed LLM reply into our command schema.

    switch_commands is forced empty: switching belongs to the environment [1].
    """
    if not isinstance(obj, dict):
        return dict(NOOP)
    mob = obj.get("mobile_commands") or {}
    if not isinstance(mob, dict):
        mob = {}
    clean: Dict[str, Any] = {}
    for name, spec in mob.items():
        if isinstance(spec, dict):
            clean[str(name)] = {k: v for k, v in spec.items()
                                if not str(k).startswith("_")}
    return {"switch_commands": {}, "mobile_commands": clean}


def _unit_states(state: Dict[str, Any]) -> Dict[str, str]:
    return {str(n): str(u.get("state", "")).strip().lower()
            for n, u in (state.get("objects") or {}).items()}


# --------------------------------------------------------------------- #
class LARAPlanner:
    """LARA: sample k plans, dedupe, rollout-select, execute until idle."""

    def __init__(self, api, llm: Optional[LLMClient], topo,
                 k: int = 3, temperature: float = 0.2,
                 switch_budget: int = 40,
                 lambda_phi: float = LAMBDA_PHI,
                 max_plan_age: int = MAX_PLAN_AGE,
                 logger=None):
        self.api = api
        self.llm = llm
        self.topo = topo
        self.k = max(1, int(k))
        self.temperature = float(temperature)
        self.lambda_phi = float(lambda_phi)
        self.max_plan_age = int(max_plan_age)
        self.switcher = SwitchOperator(api, budget=switch_budget)
        self.logger = logger

        # per-episode state
        self._active: Optional[Dict[str, Any]] = None
        self._prev_states: Dict[str, str] = {}
        self._plan_age: int = 0
        self.latencies: List[float] = []
        self.decision_points: int = 0
        self.total_plans: int = 0
        self.dedup_dropped: int = 0

        # scratch, set at the top of act() and read by _rollout()
        self._state: Dict[str, Any] = {}
        self._aff: Dict[str, Any] = {}
        self._base_phi: float = 0.0
        self._base_w: float = 0.0

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear per-episode memory. Call between episodes."""
        self._active = None
        self._prev_states = {}
        self._plan_age = 0
        self.latencies = []
        self.decision_points = 0
        self.total_plans = 0
        self.dedup_dropped = 0
        self.switcher.reset()

    # ------------------------------------------------------------------ #
    def needs_reassignment(self, state: Dict[str, Any],
                           aff: Dict[str, Any]) -> bool:
        """Paper Sec. 2.6: replan when an agent BECOMES idle or an action
        COMPLETES.

        The earlier version tested `state == "Idle"`, which is permanently true
        for an MPS holding position or a crew with no resources left, so it
        replanned every step. We now trigger only on a TRANSITION, plus two
        genuine new-decision cases, plus a staleness valve.
        """
        cur = _unit_states(state)
        prev, self._prev_states = self._prev_states, cur

        if not prev:
            return True                        # first step of the episode

        for name, s in cur.items():
            was = prev.get(name)
            if was is None:
                return True                    # new unit appeared
            if s == "idle" and was != "idle":
                return True                    # BECAME idle
            if was == "moving" and s != "moving":
                return True                    # movement finished
            if was == "repairing" and s != "repairing":
                return True                    # repair finished
            if was == "analyzing" and s != "analyzing":
                return True                    # analysis lock released

        # A crew that just arrived at a repairable fault is a new decision.
        for name, u in (state.get("objects") or {}).items():
            if u.get("type") != "RC":
                continue
            if (u.get("repairable_faults") or []) and prev.get(str(name)) == "moving":
                return True

        # An MPS whose supply just became sustainable is a new decision: the
        # horizon gate has opened and starting now is the whole ballgame.
        for name, plan in (aff.get("supply_plans") or {}).items():
            if (plan.get("feasible_now") and plan.get("sustainable")
                    and not plan.get("supplying")):
                return True

        self._plan_age += 1
        if self._plan_age >= self.max_plan_age:
            return True                        # staleness valve
        return False

    # ------------------------------------------------------------------ #
    def _propose(self, card: str) -> List[LARAPlan]:
        """Sample k plans in parallel, then deduplicate (paper Sec. 2.5)."""
        user = (f"CURRENT SITUATION (pre-validated affordance catalogue)\n"
                f"{card}\n\nProduce one JSON plan now.")
        pairs = [(LARA_SYSTEM, user)] * self.k

        try:
            outs = self.llm.complete_batch(pairs, temperature=self.temperature)
        except Exception as error:
            # FIX: the old version called recommended_command({}, {...}) here,
            # which raises on an empty state. Return an explicit empty plan and
            # let act() fall back to the validated greedy policy.
            return [LARAPlan(command=dict(NOOP),
                             reasoning=f"llm error: {error}",
                             parse_ok=False, index=0)]

        plans: List[LARAPlan] = []
        seen: set = set()
        for i, (raw, lat) in enumerate(outs):
            obj = extract_json(raw)
            cmd = _normalize(obj)
            key = _canon(cmd)
            self.total_plans += 1
            if key in seen:
                self.dedup_dropped += 1
                continue
            seen.add(key)
            plans.append(LARAPlan(
                command=cmd,
                reasoning=str((obj or {}).get("reasoning", ""))[:400],
                index=i,
                parse_ok=obj is not None,
                raw=raw,
                latency_s=lat))
        if not plans:
            plans = [LARAPlan(command=dict(NOOP),
                              reasoning="all samples failed to parse",
                              parse_ok=False, index=0)]
        return plans

    # ------------------------------------------------------------------ #
    def _rollout(self, plan: LARAPlan) -> LARAPlan:
        """Short rollout: exact one-step prediction + Phi continuation proxy.

        The greedy switch module runs first on the plan's mobile commands, so
        each candidate is scored with the switching the environment would
        actually perform [1].
        """
        mobile = plan.command.get("mobile_commands") or {}
        sw, report = self.switcher.plan(self._state, self._aff, mobile)
        plan.switch_report = report
        plan.command = {"switch_commands": sw, "mobile_commands": mobile}

        res = self.api.validate(plan.command)
        plan.sim_ok = bool(res.get("ok"))
        plan.sim_message = str(res.get("message", ""))[:300]
        if not plan.sim_ok:
            plan.rollout_score = float("-inf")
            return plan

        pred = res.get("predicted_state") or {}
        game = pred.get("game") or {}
        if game.get("locked") or (game.get("violation_loads") or []):
            plan.fatal = True
            plan.rollout_score = -1e9
            return plan

        plan.delta_weighted = (float(game.get("weighted_restored_load") or 0.0)
                               - self._base_w)
        plan.delta_potential = potential(pred) - self._base_phi
        plan.rollout_score = (plan.delta_weighted
                              + self.lambda_phi * plan.delta_potential)
        plan.rollout_steps = 1
        return plan

    # ------------------------------------------------------------------ #
    def _greedy_batch(self) -> Tuple[Dict[str, Any], bool]:
        """Validated greedy fallback. Returns (batch, sim_ok)."""
        greedy = recommended_command(self._state, self._aff, self.topo)
        sw, _ = self.switcher.plan(self._state, self._aff,
                                   greedy.get("mobile_commands") or {})
        greedy["switch_commands"] = sw
        res = self.api.validate(greedy)
        if res.get("ok"):
            return greedy, True
        return dict(NOOP), False

    # ------------------------------------------------------------------ #
    def act(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Return a dict describing this step's decision.

        Keys: batch, source, llm_calls, n_plans, n_viable, replanned,
              terminal_fatal, reasoning, decision_latency_s, chosen_index.
        """
        self._state = state
        self._aff = build_affordances(state)
        self._base_phi = potential(state)
        self._base_w = float((state.get("game") or {})
                             .get("weighted_restored_load") or 0.0)
        t = int((state.get("game") or {}).get("time_step") or 0)

        # ---- terminal-fatal guard -------------------------------------
        # If an already-Supplying MPS cannot sustain one more step, every
        # command loses restored load. Do not replan; accept the loss so the
        # runner can advance the clock.
        term = self._aff.get("terminal") or {}
        if term.get("terminal_fatal") and not term.get("already_locked"):
            batch, ok = self._greedy_batch()
            self._active = None
            self._plan_age = 0
            return {"batch": batch,
                    "source": "terminal_fatal" if ok else "terminal_fatal_noop",
                    "llm_calls": 0, "n_plans": 0, "n_viable": 0,
                    "replanned": False, "terminal_fatal": True,
                    "reasoning": str(term.get("summary", ""))[:400],
                    "decision_latency_s": None, "chosen_index": None}

        # ---- continue the committed plan unless a trigger fired --------
        replan = self.needs_reassignment(state, self._aff)
        if self._active is not None and not replan:
            mobile = {n: {"type": "continue"}
                      for n in (state.get("objects") or {})}
            sw, _ = self.switcher.plan(state, self._aff, mobile)
            return {"batch": {"switch_commands": sw, "mobile_commands": mobile},
                    "source": "lara_continue", "llm_calls": 0,
                    "n_plans": 0, "n_viable": 0, "replanned": False,
                    "terminal_fatal": False, "reasoning": "",
                    "decision_latency_s": None, "chosen_index": None}

        # ---- no LLM available: greedy ---------------------------------
        if self.llm is None:
            batch, _ok = self._greedy_batch()
            self._active = batch
            self._plan_age = 0
            return {"batch": batch, "source": "lara_noLLM", "llm_calls": 0,
                    "n_plans": 0, "n_viable": 0, "replanned": True,
                    "terminal_fatal": False, "reasoning": "",
                    "decision_latency_s": None, "chosen_index": None}

        # ---- decision point: sample k, dedupe, rollout-select ---------
        card = situation_card(state, self._aff, self.topo)
        t0 = time.time()
        plans = self._propose(card)
        plans = [self._rollout(p) for p in plans]
        latency = time.time() - t0

        self.latencies.append(latency)
        self.decision_points += 1
        self._plan_age = 0

        viable = [p for p in plans if p.sim_ok and not p.fatal]
        if viable:
            # ties resolved in favour of EARLIER proposals (paper Sec. 2.6)
            best = max(viable, key=lambda p: (p.rollout_score, -p.index))
            source = f"lara_k{len(plans)}"
        else:
            batch, ok = self._greedy_batch()
            best = LARAPlan(command=batch, index=-1, sim_ok=ok,
                            reasoning="all LLM plans rejected or fatal; "
                                      "greedy fallback")
            source = "lara_fallback"

        self._active = best.command
        llm_calls = sum(1 for p in plans if p.raw)

        if self.logger:
            self.logger.log(
                stage="lara", t=t,
                n_sampled=self.k, n_plans=len(plans),
                n_viable=len(viable), dedup_dropped=self.dedup_dropped,
                chosen_index=best.index, source=source,
                rollout_score=best.rollout_score,
                delta_weighted=best.delta_weighted,
                delta_potential=best.delta_potential,
                reasoning=best.reasoning,
                latency_s=round(latency, 2),
                switch_report=best.switch_report,
                plans=[{"i": p.index, "score": p.rollout_score,
                        "dW": p.delta_weighted, "dPhi": p.delta_potential,
                        "sim_ok": p.sim_ok, "fatal": p.fatal,
                        "parse_ok": p.parse_ok,
                        "msg": p.sim_message,
                        "cmd": p.command} for p in plans])

        return {"batch": best.command, "source": source,
                "llm_calls": llm_calls, "n_plans": len(plans),
                "n_viable": len(viable), "replanned": True,
                "terminal_fatal": False,
                "reasoning": best.reasoning,
                "decision_latency_s": latency,
                "chosen_index": best.index}

    # ------------------------------------------------------------------ #
    def stats(self) -> Dict[str, Any]:
        """Metrics matching the paper's definitions.

        `median_decision_latency_s` is the median wall-clock time to compute
        actions at a SINGLE decision point, which is the quantity that must
        stay under the restoration timestep duration [1].
        """
        return {
            "decision_points": self.decision_points,
            "median_decision_latency_s":
                round(statistics.median(self.latencies), 2)
                if self.latencies else 0.0,
            "mean_decision_latency_s":
                round(statistics.mean(self.latencies), 2)
                if self.latencies else 0.0,
            "max_decision_latency_s":
                round(max(self.latencies), 2) if self.latencies else 0.0,
            "total_plans_sampled": self.total_plans,
            "dedup_dropped": self.dedup_dropped,
            "dedup_rate": (round(self.dedup_dropped / self.total_plans, 3)
                           if self.total_plans else 0.0),
        }