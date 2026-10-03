"""Bounded multi-turn multi-agent dialogue for one simulator time step.

Rounds
  0  Dispatcher    : digest + per-unit intents                  (1 LLM call)
  1  Units         : thought + proposal (+ peer request)         (1 BATCHED call)
  2  Negotiation   : conflicted units AND units addressed by a
                     peer request get a second turn             (<=1 batched call)
  3  Dispatcher    : deterministic assembly                     (0 LLM calls)
  3b SwitchOperator: greedy validated switch toggles            (0 LLM calls)
  4  SafetyOfficer : simulator validate (hard) + LLM critique   (<=1 LLM call)
  5  Execute, then Reflector on failure/loop/fallback           (<=1 LLM call)

TERMINAL-FATAL HANDLING (fixes the observed T15 infinite loop)
--------------------------------------------------------------
The continuity rule makes some states genuinely unwinnable. Concretely: once an
MPS IS Supplying, the simulator's only legal command for it is `continue`
(moving or staying disconnects it immediately). If its stored energy has fallen
below its current P output, the simulator auto-disconnects it on the LAST FRAME
of the step no matter what we command. Its island goes dark and the score locks
to zero. EVERY action is fatal.

Previously the dialogue kept retrying such a step (chosen stayed None -> fallback
also fatal -> api.execute rejected -> the runner re-entered step() with an
identical state), producing thousands of identical `T15 ... dW=-10440` lines and
never advancing the clock.

This file now:
  * detects the condition up front via affordances["terminal"] and returns
    immediately with StepResult.terminal_fatal = True;
  * never leaves the fallback chain without a batch, even when every option is
    fatal (source="fallback_forced_fatal");
so the runner can accept the unavoidable loss and advance the clock.

COLLABORATION INSTRUMENTATION
-----------------------------
To answer "are the agents actually collaborating?" we record, per message:
  * influenced_by : which teammate's message changed this agent's action
  * yielded       : the target this agent gave up
  * reply kind    : a unit answering a peer request (not just a conflict)
and we invite a second turn when a peer ASKS something, not only when the
deterministic conflict detector fires. Volume of talk is not collaboration;
influence is. See conversation.py for the scoring.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_agents.affordances import (build_affordances, recommend,
                                    recommended_command, situation_card,
                                    team_digest, unit_block)
from llm_agents.blackboard import Blackboard
from llm_agents.client import LLMClient, extract_json
from llm_agents.memory import (LoopDetector, ReflectionMemory, SkillLibrary,
                               TrajectoryMemory, situation_signature)
from llm_agents.metrics import potential, potential_delta
from llm_agents.protocol import (K_ASSIGNMENT, K_DIGEST, K_EXECUTION,
                                 K_OBJECTION, K_PROPOSAL, K_REFLECTION,
                                 K_REQUEST, K_SWITCHPLAN, K_VERDICT, Message,
                                 ROLE_DISPATCHER, ROLE_ENV, ROLE_REFLECTOR,
                                 ROLE_SAFETY, ROLE_SWITCH, ROLE_UNIT,
                                 conflicted_units, detect_conflicts,
                                 target_key)
from llm_agents.roles import (DISPATCHER_PLAN_SYSTEM, REFLECTOR_SYSTEM,
                              SAFETY_SYSTEM, parse_dispatch_plan, parse_unit,
                              parse_verdict, unit_system, unit_user)
from llm_agents.switching import SwitchOperator

NOOP = {"switch_commands": {}, "mobile_commands": {}}

# message kinds added by this module (not in protocol.py's canonical set)
K_REPLY = "reply"
K_YIELD = "yield"


@dataclass
class DialogueConfig:
    use_dispatcher: bool = True
    use_broadcast: bool = True        # ablation: can units see teammates?
    use_negotiation: bool = True
    use_llm_critic: bool = True
    use_reflection: bool = True
    use_skills: bool = True
    use_switch_operator: bool = True
    max_revisions: int = 1
    temperature_unit: float = 0.3
    temperature_dispatch: float = 0.2
    temperature_critic: float = 0.1
    switch_budget: int = 40


@dataclass
class StepResult:
    batch: Dict[str, Any] = field(default_factory=lambda: dict(NOOP))
    source: str = "dialogue"
    sim_ok: bool = False
    sim_message: str = ""
    delta_weighted: float = 0.0
    delta_potential: float = 0.0
    lost_loads: List[str] = field(default_factory=list)
    verdict: Dict[str, Any] = field(default_factory=dict)
    rounds: int = 0
    llm_calls: int = 0
    revisions: int = 0
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    conflicts_after: List[Dict[str, Any]] = field(default_factory=list)
    safety_catches: int = 0
    proposals: Dict[str, Any] = field(default_factory=dict)
    accepted: Dict[str, bool] = field(default_factory=dict)
    # collaboration telemetry
    requests: int = 0
    replies: int = 0
    objections: int = 0
    yields: int = 0
    influenced: int = 0
    switch_report: List[str] = field(default_factory=list)
    # terminal-state telemetry
    terminal_fatal: bool = False


# --------------------------------------------------------------------- #
def _sanitize(cmd: Dict[str, Any], menu: List[dict],
              move_targets: List[dict], topo) -> Tuple[Dict[str, Any], bool]:
    """Force a proposal into the legal action space.

    Returns (command, was_repaired). `was_repaired` is the hallucination
    counter reported as `safety_catches`.
    """
    if not isinstance(cmd, dict):
        return {"type": "continue"}, True

    ctype = str(cmd.get("type", "continue")).strip().lower()
    if ctype == "repaire":
        ctype = "repair"
    allowed = {str(o.get("type")) for o in menu}

    # locked / supplying: only continue exists
    if allowed == {"continue"}:
        return {"type": "continue"}, ctype not in ("continue", "keep", "")

    if ctype == "move":
        wps = cmd.get("waypoints") or []
        clean: List[List[int]] = []
        for p in wps[:5]:
            try:
                clean.append(topo.clamp((int(p[0]), int(p[1]))))
            except Exception:
                continue
        if not clean and move_targets:
            return {"type": "move", "waypoints": [move_targets[0]["waypoint"]]}, True
        if not clean:
            return {"type": "continue"}, True
        orig: List[List[int]] = []
        for p in wps[:5]:
            try:
                orig.append([int(p[0]), int(p[1])])
            except Exception:
                pass
        return {"type": "move", "waypoints": clean}, clean != orig

    if ctype == "repair":
        for o in menu:
            if o.get("type") == "repair" and o.get("fault") == cmd.get("fault"):
                try:
                    amt = int(cmd.get("amount") or o["amount"])
                except (TypeError, ValueError):
                    amt = int(o["amount"])
                amt = max(1, min(amt, int(o["amount"])))
                return ({"type": "repair", "fault": o["fault"], "amount": amt},
                        amt != cmd.get("amount"))
        return {"type": "continue"}, True

    if ctype == "supply":
        for o in menu:
            if o.get("type") == "supply":
                try:
                    same = abs(float(cmd.get("p", -1)) - float(o["p"])) < 1e-6
                except (TypeError, ValueError):
                    same = False
                return dict(o), not same
        return {"type": "continue"}, True

    if ctype in allowed:
        return {"type": ctype}, False
    return {"type": "continue"}, True


def _detect_influence(thought: str, peers: List[str],
                      me: str) -> Optional[str]:
    """Which teammate does this thought explicitly reference?

    Cheap substring match on teammate names. It over-counts (naming a peer is
    not the same as being influenced by it) and under-counts (paraphrase without
    naming), so treat it as a screening signal; the rendered transcript is the
    ground truth.
    """
    if not thought:
        return None
    low = thought.lower()
    hits = [p for p in peers if p != me and p.lower() in low]
    return ", ".join(hits) if hits else None


# --------------------------------------------------------------------- #
class MultiAgentDialogue:
    def __init__(self, api, llm: Optional[LLMClient], topo,
                 config: DialogueConfig,
                 blackboard: Blackboard,
                 reflections: Optional[ReflectionMemory] = None,
                 skills: Optional[SkillLibrary] = None):
        self.api = api
        self.llm = llm
        self.topo = topo
        self.cfg = config
        self.bb = blackboard
        self.reflections = reflections or ReflectionMemory()
        self.skills = skills or SkillLibrary()
        self.trajectory = TrajectoryMemory()
        self.loops = LoopDetector()
        self.switcher = SwitchOperator(api, budget=config.switch_budget)

    # ------------------------------------------------------------ helpers
    def _post(self, **kw) -> Message:
        return self.bb.post(Message(**kw))

    def _ask(self, system: str, user: str, temperature: float):
        raw, lat = self.llm.complete_timed(system, user, temperature=temperature)
        return extract_json(raw), raw, lat

    def _ask_batch(self, pairs: List[Tuple[str, str]], temperature: float):
        outs = self.llm.complete_batch(pairs, temperature=temperature)
        return [(extract_json(raw), raw, lat) for raw, lat in outs]

    def reset(self) -> None:
        """Clear per-episode state. Call between episodes."""
        self.trajectory = TrajectoryMemory()
        self.loops = LoopDetector()
        self.switcher.reset()

    # ------------------------------------------------------------ main
    def step(self, state: Dict[str, Any]) -> StepResult:
        t = int((state.get("game") or {}).get("time_step") or 0)
        aff = build_affordances(state)
        recs = recommend(state, aff, self.topo)
        digest = team_digest(state, aff)
        card = situation_card(state, aff, self.topo)
        sig = situation_signature(state, aff)
        base_phi = potential(state)

        res = StepResult()
        units = sorted((state.get("objects") or {}).keys())

        # ------------------------------------------------------------------
        # TERMINAL GUARD. If an already-Supplying MPS cannot sustain even one
        # more step, EVERY command loses previously energized load. Retrying is
        # pointless and loops forever. Accept the loss, execute the cheapest
        # validated batch, and let the runner advance the clock.
        # ------------------------------------------------------------------
        term = aff.get("terminal") or {}
        if term.get("terminal_fatal") and not term.get("already_locked"):
            res.terminal_fatal = True
            res.sim_message = str(term.get("summary", ""))[:400]

            greedy = recommended_command(state, aff, self.topo)
            vres = self.api.validate(greedy)
            res.sim_ok = bool(vres.get("ok"))
            pred = (vres.get("predicted_state") or {}).get("game") or {}
            res.lost_loads = list(pred.get("violation_loads") or [])
            if res.sim_ok:
                res.batch, res.source = greedy, "terminal_fatal_greedy"
            else:
                res.batch, res.source = dict(NOOP), "terminal_fatal_noop"

            self._post(t=t, round=4, speaker="SafetyOfficer", role=ROLE_SAFETY,
                       kind=K_VERDICT,
                       text="UNRECOVERABLE: " + str(term.get("summary", "")),
                       payload={"verdict": "terminal", "terminal_fatal": True,
                                "doomed_units": term.get("doomed_units"),
                                "lost_loads": res.lost_loads})
            self._post(t=t, round=4, speaker="Dispatcher", role=ROLE_DISPATCHER,
                       kind=K_ASSIGNMENT,
                       text="accepting unavoidable loss and advancing the clock",
                       payload=dict(res.batch))
            return res

        # ---------------- round 0: dispatcher intents
        intents: Dict[str, str] = {}
        if self.llm and self.cfg.use_dispatcher:
            user = (f"SITUATION\n{card}\n\nLESSONS FROM PAST EPISODES\n"
                    f"{self.reflections.render()}\n\nRECENT TEAM HISTORY\n"
                    f"{self.trajectory.render()}\n\nAssign intents now.")
            obj, raw, lat = self._ask(DISPATCHER_PLAN_SYSTEM, user,
                                      self.cfg.temperature_dispatch)
            plan = parse_dispatch_plan(obj, units)
            intents = plan["intents"]
            res.llm_calls += 1
            self._post(t=t, round=0, speaker="Dispatcher", role=ROLE_DISPATCHER,
                       kind=K_DIGEST,
                       text=plan["situation"] +
                            (" RISK: " + plan["risk"] if plan["risk"] else ""),
                       payload={"intents": intents}, raw=raw, latency_s=lat,
                       prompt_chars=len(user), response_chars=len(raw),
                       parse_ok=obj is not None)
        else:
            self._post(t=t, round=0, speaker="Environment", role=ROLE_ENV,
                       kind=K_DIGEST, text=digest)

        feedback = ""
        chosen: Optional[Dict[str, Any]] = None

        for attempt in range(self.cfg.max_revisions + 1):
            res.revisions = attempt

            # ---------------- round 1: unit proposals (batched)
            proposals: Dict[str, Dict[str, Any]] = {}
            pending_requests: Dict[str, List[str]] = {}   # target -> [askers]

            if self.llm:
                pairs, order = [], []
                for name in units:
                    u = (state.get("objects") or {})[name]
                    menu = aff["object_options"].get(name, [])
                    if len(menu) == 1:      # locked / supplying: no call needed
                        proposals[name] = {
                            "thought": "locked: only continue is legal",
                            "command": {"type": "continue"},
                            "request": None, "confidence": 1.0,
                            "influenced_by": None}
                        continue
                    feed = (self.bb.render_feed(name, t=t)
                            if self.cfg.use_broadcast
                            else "(communication disabled in this configuration)")
                    skills_txt = (self.skills.render(sig)
                                  if self.cfg.use_skills else "(disabled)")
                    rec_cmd, rec_why = recs.get(name, ({"type": "continue"}, ""))
                    pairs.append((
                        unit_system(name, u.get("type", "RC")),
                        unit_user(name, digest,
                                  unit_block(state, aff, name), feed,
                                  intents.get(name, ""),
                                  f"{json.dumps(rec_cmd)}   <- {rec_why}",
                                  self.reflections.render()
                                  + "\nCOMPARABLE PAST PLANS\n" + skills_txt,
                                  self.trajectory.render(), feedback)))
                    order.append(name)

                if pairs:
                    outs = self._ask_batch(pairs, self.cfg.temperature_unit)
                    for name, (obj, raw, lat) in zip(order, outs):
                        parsed = parse_unit(obj)
                        infl = (_detect_influence(parsed["thought"], units, name)
                                if self.cfg.use_broadcast else None)
                        parsed["influenced_by"] = infl
                        proposals[name] = parsed
                        res.llm_calls += 1
                        if infl:
                            res.influenced += 1
                        self._post(t=t, round=1, speaker=name, role=ROLE_UNIT,
                                   kind=K_PROPOSAL, text=parsed["thought"],
                                   payload={"command": parsed["command"],
                                            "confidence": parsed["confidence"],
                                            "influenced_by": infl},
                                   raw=raw, latency_s=lat,
                                   response_chars=len(raw),
                                   parse_ok=obj is not None)
                        if parsed["request"] and self.cfg.use_broadcast:
                            tgt = str(parsed["request"]["to"])
                            res.requests += 1
                            pending_requests.setdefault(tgt, []).append(name)
                            self._post(t=t, round=1, speaker=name,
                                       role=ROLE_UNIT, kind=K_REQUEST,
                                       to=[tgt],
                                       text=str(parsed["request"].get("text", ""))[:300])
            else:
                for name, (cmd, why) in recs.items():
                    proposals[name] = {"thought": why, "command": cmd,
                                       "request": None, "confidence": 1.0,
                                       "influenced_by": None}

            # sanitize + record acceptance / hallucination catches
            for name, p in proposals.items():
                cmd, repaired = _sanitize(p["command"],
                                          aff["object_options"].get(name, []),
                                          aff["move_targets"].get(name, []),
                                          self.topo)
                res.accepted[name] = not repaired
                res.safety_catches += int(repaired)
                p["command"] = cmd
            res.proposals = {n: p["command"] for n, p in proposals.items()}
            pre_targets = {n: target_key(c) for n, c in res.proposals.items()}

            # ---------------- round 2: negotiation
            # Speak again if (a) you are in a detected conflict, OR
            # (b) a teammate addressed a request to you. (b) is what makes the
            # dialogue a conversation rather than a broadcast.
            conflicts = detect_conflicts(res.proposals, state)
            res.conflicts = conflicts
            if self.llm and self.cfg.use_negotiation and self.cfg.use_broadcast:
                speak: Set[str] = set(conflicted_units(conflicts))
                speak |= set(pending_requests.keys())
                speak &= set(proposals.keys())
                speak = {n for n in speak
                         if len(aff["object_options"].get(n, [])) > 1}

                if speak:
                    detail = "\n".join(f"- {c['kind']}: {c['detail']}"
                                       for c in conflicts) or "(no hard conflict)"
                    pairs, order, why_called = [], [], {}
                    conf_units = set(conflicted_units(conflicts))
                    for name in sorted(speak):
                        asks = pending_requests.get(name) or []
                        in_conf = name in conf_units
                        why_called[name] = ("conflict" if in_conf else "asked")
                        extra = ["CONFLICTS DETECTED BY THE TEAM:", detail]
                        if asks:
                            extra.append(
                                f"\n{', '.join(asks)} addressed a request to you "
                                f"(see TEAM MESSAGES). Answer it explicitly: say "
                                f"whether you accept, and name the teammate.")
                        extra.append(
                            "\nEither yield to a teammate and pick a different "
                            "target, or justify keeping yours in one sentence. "
                            "Name the teammate you are responding to.")
                        rec_cmd, rec_why = recs.get(name, ({"type": "continue"}, ""))
                        pairs.append((
                            unit_system(name,
                                        (state["objects"][name]).get("type", "RC")),
                            unit_user(name, digest,
                                      unit_block(state, aff, name),
                                      self.bb.render_feed(name, t=t),
                                      intents.get(name, ""),
                                      f"{json.dumps(rec_cmd)}   <- {rec_why}",
                                      self.reflections.render(),
                                      self.trajectory.render(),
                                      "\n".join(extra))))
                        order.append(name)

                    outs = self._ask_batch(pairs, self.cfg.temperature_unit)
                    for name, (obj, raw, lat) in zip(order, outs):
                        parsed = parse_unit(obj)
                        cmd, repaired = _sanitize(
                            parsed["command"],
                            aff["object_options"].get(name, []),
                            aff["move_targets"].get(name, []), self.topo)
                        infl = _detect_influence(parsed["thought"], units, name)
                        changed = (target_key(cmd) != pre_targets.get(name))
                        yielded = None
                        if changed and pre_targets.get(name):
                            yielded = str(pre_targets[name])

                        proposals[name]["command"] = cmd
                        proposals[name]["thought"] = parsed["thought"]
                        proposals[name]["influenced_by"] = infl
                        res.proposals[name] = cmd
                        res.safety_catches += int(repaired)
                        res.llm_calls += 1
                        if infl:
                            res.influenced += 1

                        if yielded:
                            kind, res.yields = K_YIELD, res.yields + 1
                        elif why_called[name] == "conflict":
                            kind, res.objections = K_OBJECTION, res.objections + 1
                        else:
                            kind, res.replies = K_REPLY, res.replies + 1

                        to = (pending_requests.get(name) or ["ALL"])
                        self._post(t=t, round=2, speaker=name, role=ROLE_UNIT,
                                   kind=kind, to=list(to),
                                   text=parsed["thought"],
                                   payload={"command": cmd,
                                            "confidence": parsed["confidence"],
                                            "influenced_by": infl,
                                            "yielded": yielded,
                                            "called_because": why_called[name]},
                                   raw=raw, latency_s=lat,
                                   parse_ok=obj is not None)
                res.conflicts_after = detect_conflicts(res.proposals, state)
            else:
                res.conflicts_after = list(conflicts)

            # ---------------- round 3: deterministic assembly
            mobile = dict(res.proposals)
            for c in detect_conflicts(mobile, state):
                if c["kind"] == "duplicate_target":
                    keep = max(c["units"],
                               key=lambda n: proposals.get(n, {}).get("confidence", 0.0))
                    for n in c["units"]:
                        if n == keep:
                            continue
                        alt = aff["move_targets"].get(n) or []
                        mobile[n] = ({"type": "move",
                                      "waypoints": [alt[1]["waypoint"]]}
                                     if len(alt) > 1 else {"type": "continue"})
                        res.safety_catches += 1
                elif c["kind"] in ("analysis_lock_violation",
                                   "supplying_mps_disturbed"):
                    for n in c["units"]:
                        mobile[n] = {"type": "continue"}
                        res.safety_catches += 1

            # ---------------- round 3b: switch operator (NO LLM)
            switches: Dict[str, str] = {}
            if self.cfg.use_switch_operator:
                switches, report = self.switcher.plan(state, aff, mobile)
                res.switch_report = report
                self._post(t=t, round=3, speaker="SwitchOperator",
                           role=ROLE_SWITCH, kind=K_SWITCHPLAN,
                           text="; ".join(report),
                           payload={"switch_commands": switches})

            batch = {"switch_commands": switches, "mobile_commands": mobile}
            self._post(t=t, round=3, speaker="Dispatcher", role=ROLE_DISPATCHER,
                       kind=K_ASSIGNMENT, text="joint batch assembled",
                       payload={**batch,
                                "n_conflicts": len(res.conflicts),
                                "n_conflicts_after": len(res.conflicts_after),
                                "peer_influenced": res.influenced})

            # ---------------- round 4: hard gate + critic
            vres = self.api.validate(batch)
            res.sim_ok = bool(vres.get("ok"))
            res.sim_message = str(vres.get("message", ""))[:400]
            pred = vres.get("predicted_state") or {}
            pgame = pred.get("game") or {}
            res.lost_loads = list(pgame.get("violation_loads") or [])
            fatal = bool(pgame.get("locked")) or bool(res.lost_loads)
            before_w = float((state.get("game") or {}).get("weighted_restored_load") or 0.0)
            res.delta_weighted = float(pgame.get("weighted_restored_load") or 0.0) - before_w
            res.delta_potential = potential_delta(pred, base_phi) if pred else 0.0

            verdict = {"verdict": "accept", "risk": "low", "score": 5.0,
                       "reasons": [], "suggestion": ""}
            if self.llm and self.cfg.use_llm_critic and res.sim_ok and not fatal:
                user = (f"SITUATION\n{digest}\n\nPROPOSAL\n{json.dumps(batch)}\n\n"
                        f"SIMULATOR PREDICTION (ground truth)\n"
                        f"  next |P|*W: {pgame.get('weighted_restored_load')}\n"
                        f"  delta vs now: {res.delta_weighted:+g}\n"
                        f"  delta progress potential: {res.delta_potential:+g}\n"
                        f"  energized next: {pgame.get('energized_loads')}\n"
                        f"  lost loads: {res.lost_loads}\n"
                        f"  score locked: {pgame.get('locked')}\n\n"
                        "Return the JSON verdict now.")
                obj, raw, lat = self._ask(SAFETY_SYSTEM, user,
                                          self.cfg.temperature_critic)
                verdict = parse_verdict(obj)
                res.llm_calls += 1
                self._post(t=t, round=4, speaker="SafetyOfficer",
                           role=ROLE_SAFETY, kind=K_VERDICT,
                           text="; ".join(str(r) for r in verdict["reasons"]),
                           payload={k: verdict[k] for k in
                                    ("verdict", "risk", "score", "suggestion")},
                           raw=raw, latency_s=lat, parse_ok=obj is not None)
            elif (not res.sim_ok) or fatal:
                verdict = {"verdict": "reject", "risk": "high", "score": 0.0,
                           "reasons": [res.sim_message
                                       or f"fatal: loses {res.lost_loads}"],
                           "suggestion": "Do not de-energize restored load; "
                                         "never move a Supplying MPS."}
                self._post(t=t, round=4, speaker="SafetyOfficer",
                           role=ROLE_SAFETY, kind=K_VERDICT,
                           text=verdict["reasons"][0],
                           payload={"verdict": "reject", "sim_ok": res.sim_ok,
                                    "lost_loads": res.lost_loads,
                                    "conflicts": res.conflicts})
            res.verdict = verdict
            res.rounds = 4

            if res.sim_ok and not fatal and verdict["verdict"] != "reject":
                chosen = batch
                res.source = f"dialogue_rev{attempt}"
                break

            feedback = (f"REJECTED. simulator says: {res.sim_message or 'n/a'}. "
                        f"lost loads: {res.lost_loads}. "
                        f"critic suggestion: {verdict.get('suggestion', '')}")

        # ---------------- fallback chain (never execute unvalidated)
        if chosen is None:
            greedy = recommended_command(state, aff, self.topo)
            if self.cfg.use_switch_operator:
                sw, _ = self.switcher.plan(state, aff, greedy["mobile_commands"])
                greedy["switch_commands"] = sw
            g = self.api.validate(greedy)
            gg = (g.get("predicted_state") or {}).get("game") or {}
            if g.get("ok") and not gg.get("locked") \
               and not (gg.get("violation_loads") or []):
                chosen, res.source = greedy, "fallback_greedy"
            else:
                noop = dict(NOOP)
                nres = self.api.validate(noop)
                ng = (nres.get("predicted_state") or {}).get("game") or {}
                if nres.get("ok") and not ng.get("locked") \
                   and not (ng.get("violation_loads") or []):
                    chosen, res.source = noop, "fallback_noop"
                else:
                    # EVERY option is fatal. Do NOT retry -- that is the loop.
                    # Take greedy if the simulator accepts it at all, else no-op,
                    # and flag the step so the runner records a dead state.
                    chosen = greedy if g.get("ok") else noop
                    res.source = "fallback_forced_fatal"
                    res.terminal_fatal = True
                    self._post(t=t, round=4, speaker="SafetyOfficer",
                               role=ROLE_SAFETY, kind=K_VERDICT,
                               text="every option violates continuity; forcing "
                                    "execution to advance the clock",
                               payload={"verdict": "terminal",
                                        "terminal_fatal": True,
                                        "lost_loads": res.lost_loads})
            self._post(t=t, round=4, speaker="Dispatcher", role=ROLE_DISPATCHER,
                       kind=K_ASSIGNMENT, text=f"falling back: {res.source}",
                       payload=dict(chosen))

        res.batch = chosen
        return res

    # ------------------------------------------------------------ after execute
    def observe_execution(self, t: int, res: StepResult,
                          after_game: Dict[str, Any], ok: bool, message: str,
                          state_before: Dict[str, Any],
                          aff_before: Dict[str, Any]) -> None:
        self._post(t=t, round=5, speaker="Environment", role=ROLE_ENV,
                   kind=K_EXECUTION,
                   text=("accepted" if ok else "REJECTED: " + str(message)[:200]),
                   payload={"score": after_game.get("score"),
                            "weighted": after_game.get("weighted_restored_load"),
                            "locked": after_game.get("locked"),
                            "energized": after_game.get("energized_loads"),
                            "terminal_fatal": res.terminal_fatal})
        self.trajectory.add(t, res.batch, res.delta_weighted,
                            "" if ok else "rejected")

        loop_note = self.loops.observe(
            json.dumps(res.batch, sort_keys=True),
            json.dumps(after_game.get("energized_loads") or []))
        trigger = ((not ok) or bool(after_game.get("locked"))
                   or (loop_note is not None)
                   or res.source.startswith("fallback")
                   or res.terminal_fatal)

        if self.llm and self.cfg.use_reflection and trigger:
            user = (f"WHAT WE DID\n{json.dumps(res.batch)}\n\n"
                    f"WHAT HAPPENED\n accepted={ok}; message={str(message)[:200]}\n"
                    f" d|P|W={res.delta_weighted:+g}; "
                    f"locked={after_game.get('locked')}\n"
                    f" heuristic flag: {loop_note or 'none'}\n"
                    f" fallback used: {res.source}\n"
                    f" unrecoverable state: {res.terminal_fatal}\n\n"
                    f"TEAM HISTORY\n{self.trajectory.render()}\n\n"
                    "Write the lesson now.")
            obj, raw, lat = self._ask(REFLECTOR_SYSTEM, user, 0.3)
            lesson = str((obj or {}).get("lesson", ""))[:400]
            tags = (obj or {}).get("tags") or []
            if lesson:
                self.reflections.add(lesson, tags, {"t": t})
            self._post(t=t, round=5, speaker="Reflector", role=ROLE_REFLECTOR,
                       kind=K_REFLECTION, text=lesson, payload={"tags": tags},
                       raw=raw, latency_s=lat, parse_ok=obj is not None)

        # Voyager-style: commit only validated, non-regressing plans
        if ok and self.cfg.use_skills and not after_game.get("locked") \
           and not res.terminal_fatal \
           and (res.delta_weighted > 0 or res.delta_potential > 0):
            sig = situation_signature(state_before, aff_before)
            self.skills.add(sig, f"step {t}", res.batch,
                            res.delta_weighted, res.delta_potential)