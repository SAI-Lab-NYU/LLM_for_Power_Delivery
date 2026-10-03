"""Single-agent Generator + Verifier + Selector baselines, with full logging.

This file is the ABLATION ARM: gen_only / gen_ver / gen_ver_llm. The
multi-agent society lives in llm_agents/dialogue.py.

Change vs. the original: candidate ranking no longer relies on
delta_weighted alone. Before the transmission grid returns (step 6 by default)
and while the network is still partly unknown, delta_weighted is ZERO for most
of the episode, which made selection blind exactly when the decisions mattered.
We add a dense progress potential Phi (repair progress, analysed faults, known
network elements, energizable areas) from llm_agents/metrics.py.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_agents.client import LLMClient, extract_json
from llm_agents.metrics import potential, potential_delta
from llm_agents.sim_api import SimulatorAPI
from llm_agents.trace_log import InteractionLogger

# Weight on the dense potential relative to realised weighted load.
LAMBDA_POTENTIAL = 1.0
# Weight on the LLM critic score: tie-break only, never dominant.
LAMBDA_CRITIC = 1.0

GENERATOR_SYSTEM = """You are the Generator in a post-disaster power distribution restoration system.
Output ONE JSON command batch for exactly one simulator time step.

Objective: maximize the sum over time of sum(|P| * W) over ENERGIZED loads.
Restoring high-weight load EARLIER is worth more (it re-counts every step).
CRITICAL: if a load that was energized ever loses power, the score is
permanently locked to zero. Prefer safe, monotone restoration.

Output format EXACTLY, and nothing else:
{"switch_commands": {}, "mobile_commands": {"RC1": {"type": "continue"}}}

Rules:
- Choose ONLY from the provided legal command menu / move waypoints.
- Omitted units default to "continue".
- An analysis-locked unit accepts ONLY {"type":"continue"}.
- A Supplying MPS accepts ONLY {"type":"continue"}. Moving or staying
  disconnects it, de-energizes its island, and LOCKS the score to zero.
- Move waypoints are integer [x,y] pairs, at most 5 per command.
- MPS supply on an islanded cluster must EXACTLY match the cluster P and Q.
- Leave "switch_commands" empty: switch closures are handled automatically.
- Never create an energized loop.
- A RECOMMENDED ACTION is given for each unit. Follow it unless you have a
  concrete, stated reason to deviate.

Return JSON only. No prose, no explanation."""

VERIFIER_SYSTEM = """You are the Verifier in a post-disaster restoration system.
You get the situation, a proposed command batch, and the simulator's
ground-truth prediction of the next state. The simulator has ALREADY decided
legality; judge quality and name the single best improvement.

Judge on:
1. SAFETY: does any previously energized load lose power, or is the score
   locked? (fatal)
2. PROGRESS: does weighted restored load rise, or does this set up progress
   (fault analysed, repair advanced, unknown network pixels revealed)?
3. EFFICIENCY: are crews/MPS idle while useful work exists?
4. SEQUENCING: are units routed toward high-weight unserved load?

Reply with JSON only:
{"verdict":"accept"|"revise"|"reject","risk":"none"|"low"|"medium"|"high",
 "score":<0-10 integer>,"reasons":["..."],"suggestion":"one improvement or \"\""}"""


# --------------------------------------------------------------------- #
@dataclass
class Candidate:
    command: dict
    source: str
    sim_ok: bool = False
    sim_message: str = ""
    predicted: Optional[dict] = None
    delta_weighted: float = 0.0
    delta_potential: float = 0.0
    lost_loads: List[str] = field(default_factory=list)
    verifier: Dict[str, Any] = field(default_factory=dict)
    final_score: float = -1e9
    parse_ok: bool = True
    latency_s: float = 0.0
    raw_response: str = ""


NOOP = {"switch_commands": {}, "mobile_commands": {}}
FATAL_SCORE = -1e5


def _normalize(command: Any) -> dict:
    if not isinstance(command, dict):
        return {"switch_commands": {}, "mobile_commands": {}}
    switch = command.get("switch_commands") or {}
    mobile = command.get("mobile_commands") or {}
    if not isinstance(switch, dict):
        switch = {}
    if not isinstance(mobile, dict):
        mobile = {}
    clean_switch = {str(k): str(v).strip().lower() for k, v in switch.items()
                    if str(v).strip().lower() in ("opened", "closed")}
    clean_mobile = {}
    for name, spec in mobile.items():
        if not isinstance(spec, dict):
            continue
        spec = {k: v for k, v in spec.items() if not str(k).startswith("_")}
        clean_mobile[str(name)] = spec
    return {"switch_commands": clean_switch, "mobile_commands": clean_mobile}


# --------------------------------------------------------------------- #
class Generator:
    def __init__(self, llm: LLMClient,
                 logger: Optional[InteractionLogger] = None):
        self.llm = llm
        self.logger = logger

    def propose(self, card: str, k: int = 3, feedback: str = "",
                timestep: Optional[int] = None) -> List[Candidate]:
        """Sample k diverse batches in ONE batched forward pass, plus a no-op."""
        temperatures = [0.2, 0.5, 0.8, 0.35, 0.65]
        pairs, temps = [], []
        for index in range(max(1, k)):
            temp = temperatures[index % len(temperatures)]
            user = f"CURRENT SITUATION\n{card}\n"
            if feedback:
                user += f"\nVERIFIER FEEDBACK ON A PREVIOUS ATTEMPT:\n{feedback}\n"
            user += "\nProduce one JSON command batch now."
            pairs.append((GENERATOR_SYSTEM, user))
            temps.append(temp)

        candidates: List[Candidate] = []
        # Batch per temperature bucket so sampling temperature stays honest.
        by_temp: Dict[float, List[int]] = {}
        for i, t in enumerate(temps):
            by_temp.setdefault(t, []).append(i)

        outputs: Dict[int, Any] = {}
        for temp, idxs in by_temp.items():
            try:
                res = self.llm.complete_batch([pairs[i] for i in idxs],
                                              temperature=temp)
            except Exception as error:
                for i in idxs:
                    outputs[i] = ("", 0.0, str(error))
                if self.logger:
                    self.logger.log(stage="generator", t=timestep,
                                    temperature=temp, error=str(error))
                continue
            for i, (raw, lat) in zip(idxs, res):
                outputs[i] = (raw, lat, None)

        for index in range(len(pairs)):
            raw, lat, err = outputs.get(index, ("", 0.0, "no output"))
            if err is not None:
                candidates.append(Candidate(command=dict(NOOP),
                                            source=f"llm_error:{err}",
                                            parse_ok=False))
                continue
            parsed = extract_json(raw)
            cand = Candidate(command=_normalize(parsed),
                             source=f"llm_t{temps[index]}",
                             parse_ok=parsed is not None,
                             latency_s=lat,
                             raw_response=raw)
            if self.logger:
                self.logger.log(stage="generator", t=timestep,
                                candidate_index=index, temperature=temps[index],
                                prompt_user=pairs[index][1], raw_response=raw,
                                parse_ok=cand.parse_ok,
                                parsed_command=cand.command,
                                latency_s=round(lat, 2),
                                prompt_chars=len(pairs[index][1]),
                                response_chars=len(raw))
            candidates.append(cand)

        # safe no-op so the loop can never stall
        candidates.append(Candidate(command=dict(NOOP), source="noop_baseline"))
        return candidates


# --------------------------------------------------------------------- #
class Verifier:
    """Hard gate = simulator /api/text/validate. Soft gate = LLM critique."""

    def __init__(self, api: SimulatorAPI, llm: Optional[LLMClient] = None,
                 use_llm: bool = True,
                 logger: Optional[InteractionLogger] = None):
        self.api = api
        self.llm = llm
        self.use_llm = use_llm and llm is not None
        self.logger = logger

    def check(self, candidate: Candidate, state: dict, card: str,
              timestep: Optional[int] = None) -> Candidate:
        before = state.get("game", {}) or {}
        before_weighted = float(before.get("weighted_restored_load") or 0.0)
        before_loads = set(before.get("energized_loads") or [])
        base_phi = potential(state)

        result = self.api.validate(candidate.command)
        candidate.sim_ok = bool(result.get("ok"))
        candidate.sim_message = str(result.get("message", ""))

        if not candidate.sim_ok:
            candidate.final_score = -1e6
            if self.logger:
                self.logger.log(stage="verifier", t=timestep,
                                candidate_source=candidate.source,
                                command=candidate.command, sim_ok=False,
                                sim_message=candidate.sim_message)
            return candidate

        predicted = result.get("predicted_state", {}) or {}
        candidate.predicted = predicted
        game = predicted.get("game", {}) or {}
        after_weighted = float(game.get("weighted_restored_load") or 0.0)
        after_loads = set(game.get("energized_loads") or [])
        candidate.delta_weighted = after_weighted - before_weighted
        candidate.delta_potential = potential_delta(predicted, base_phi)

        # Trust the simulator's own violation bookkeeping: it compares against
        # its EVER-energized history, so it catches losses the naive local diff
        # cannot -- notably at T0 when before_loads is empty.
        sim_violations = list(game.get("violation_loads") or [])
        sim_locked = bool(game.get("locked"))
        local_lost = sorted(before_loads - after_loads)
        candidate.lost_loads = sim_violations or local_lost

        if sim_locked or candidate.lost_loads:
            candidate.final_score = FATAL_SCORE
            candidate.verifier = {
                "verdict": "reject", "risk": "high", "score": 0,
                "reasons": [f"score locked={sim_locked}; "
                            f"loses energized loads: {candidate.lost_loads}"],
                "suggestion": "Do not de-energize restored load. Never move or "
                              "stay a Supplying MPS."}
            if self.logger:
                self.logger.log(stage="verifier", t=timestep,
                                candidate_source=candidate.source,
                                command=candidate.command, sim_ok=True,
                                sim_message=candidate.sim_message,
                                delta_weighted=candidate.delta_weighted,
                                delta_potential=candidate.delta_potential,
                                lost_loads=candidate.lost_loads,
                                sim_locked=sim_locked,
                                llm_verdict=candidate.verifier,
                                final_score=candidate.final_score)
            return candidate

        if self.use_llm:
            candidate.verifier = self._llm_critique(candidate, card, game, timestep)
        else:
            candidate.verifier = {"verdict": "accept", "risk": "low",
                                  "score": 5, "reasons": [], "suggestion": ""}

        llm_score = float(candidate.verifier.get("score") or 5)
        candidate.final_score = (candidate.delta_weighted
                                 + LAMBDA_POTENTIAL * candidate.delta_potential
                                 + LAMBDA_CRITIC * llm_score)

        if self.logger:
            self.logger.log(stage="verifier", t=timestep,
                            candidate_source=candidate.source,
                            command=candidate.command, sim_ok=True,
                            sim_message=candidate.sim_message,
                            delta_weighted=candidate.delta_weighted,
                            delta_potential=candidate.delta_potential,
                            lost_loads=candidate.lost_loads,
                            sim_locked=sim_locked,
                            llm_verdict=candidate.verifier,
                            final_score=candidate.final_score)
        return candidate

    def _llm_critique(self, candidate: Candidate, card: str,
                      predicted_game: dict, timestep) -> dict:
        user = (f"SITUATION\n{card}\n\n"
                f"PROPOSAL\n{json.dumps(candidate.command)}\n\n"
                f"SIMULATOR PREDICTION (ground truth)\n"
                f"  accepted: {candidate.sim_ok}\n"
                f"  next |P|*W: {predicted_game.get('weighted_restored_load')}\n"
                f"  delta vs now: {candidate.delta_weighted:+g}\n"
                f"  delta progress potential: {candidate.delta_potential:+g}\n"
                f"  energized next: {predicted_game.get('energized_loads')}\n"
                f"  lost loads: {candidate.lost_loads}\n"
                f"  score locked: {predicted_game.get('locked')}\n\n"
                "Return the JSON verdict now.")
        try:
            raw, latency = self.llm.complete_timed(VERIFIER_SYSTEM, user,
                                                   temperature=0.1)
        except Exception:
            return {"verdict": "accept", "risk": "low", "score": 5,
                    "reasons": ["verifier LLM unavailable"], "suggestion": ""}

        parsed = extract_json(raw) or {}
        if self.logger:
            self.logger.log(stage="verifier_llm", t=timestep,
                            candidate_source=candidate.source,
                            prompt_user=user, raw_response=raw,
                            parse_ok=bool(parsed),
                            latency_s=round(latency, 2))
        parsed.setdefault("verdict", "accept")
        parsed.setdefault("risk", "low")
        parsed.setdefault("score", 5)
        parsed.setdefault("reasons", [])
        parsed.setdefault("suggestion", "")
        return parsed


# --------------------------------------------------------------------- #
def select(candidates: List[Candidate]) -> Candidate:
    """Prefer validated, non-fatal candidates. Never knowingly pick a
    score-locking action: fall back to the guaranteed-legal no-op."""
    viable = [c for c in candidates if c.sim_ok and c.final_score > FATAL_SCORE]
    if viable:
        return max(viable, key=lambda c: c.final_score)
    for c in candidates:
        if c.source == "noop_baseline":
            return c
    return Candidate(command=dict(NOOP), source="noop_fallback", sim_ok=True)