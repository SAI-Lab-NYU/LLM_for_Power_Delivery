# LLM-based Multi-Agent Restoration for Post-Disaster Distribution Grids

An LLM-driven multi-agent decision layer on top of the DSR V23 IEEE-13
distribution-system-restoration simulator, built as an extension of LARA
(Findings of EACL 2026) from a *single* language-conditioned planner to a
*communicating society* of role-specialised agents with grounded verification
and verbal memory.

Everything here runs on open-weight models (Qwen2.5 / Llama-3.1 / Mistral
class). No API keys, no external services. The simulator is the only arbiter of
legality.

---

## 1. Problem, in one paragraph

At each time step you command Repair Crews (RC), a Patrol & Assessment Crew
(PAC), a Mobile Power Source (MPS) and a set of switches. Score is
`sum over t of sum_j |P_j| * W_j` over loads energised at step `t`. Restoring a
high-weight load *earlier* is worth more, because it re-counts at every later
step. **If any load that was once energised ever loses power, the score is
permanently locked to zero.** Loads cannot be switched on directly; they
energise automatically once their Area is connected to a live source, and an
Area only becomes connectable once it is fully *known* and *fault-free*. That
produces a hard dependency chain:
PATROL (reveal unknown node/line pixels)
-> ANALYSE (one mandatory locked step at the fault)
-> REPAIR (crew standing exactly on the fault pixel)
-> RECONNECT (switch closure into a known fault-free area,
or MPS supplying an island's EXACT P and Q)

A 7B model will not infer that chain from a raw JSON state dump. The whole
design of this package is about making the chain *legible and pre-validated*.

---

## 2. Three non-negotiable invariants

1. **The simulator is the only arbiter of legality.** Every candidate batch
   goes through `POST /api/text/validate` before `POST /api/text/execute`. The
   LLM's verdict is only a tie-breaker. This is what makes the safety guarantee
   real rather than merely prompted — the same separation LARA reports as
   yielding zero constraint violations.
2. **The fallback must itself be validated.** Because monotonicity loss is
   fatal, the chain is: best validated dialogue batch → validated greedy batch
   → no-op. We never execute an unvalidated batch.
3. **The LLM can only help, never hurt.** Given (1)+(2),
   `score(multi-agent) >= score(greedy) - eps`. The *delta over greedy* is
   therefore exactly the measured contribution of language-conditioned
   coordination, which is the number the paper is about.

---

## 3. Architecture
┌──────────────── BLACKBOARD (append-only message pool) ────────────────┐
│ every utterance: {t, round, speaker, role, kind, to[], text, payload} │
└───────────────────────────────────────────────────────────────────────┘
/api/text/state ──digest──▶ Dispatcher ──intents──▶ RC1 RC2 PAC1 MPS1 ──proposals──▶ SafetyOfficer
▲ │ ▲ │
└──── negotiation ─────┘ │ │
│ │ ▼
deterministic assembly ◀────────────┘ /api/text/validate
│ (HARD GATE, truth)
▼
SwitchOperator (greedy, NO LLM) ──▶ /api/text/execute ──▶ Reflector


| Component | What it does | Provenance |
|---|---|---|
| **Affordance tables** (`affordances.py`) | vetted per-unit menus; no ID can be hallucinated | LARA's affordance-first design: decouple feasibility computation from LLM reasoning |
| **SwitchOperator** (`switching.py`) | greedy switch reconfiguration, deterministic, zero LLM calls; marginal benefit measured by the simulator's own one-step prediction | LARA keeps switching in the *environment* because `2^|S|` search with radiality is not something a 7B can execute |
| **Dispatcher** (`roles.py`, `dialogue.py`) | one-line intent per unit; then deterministic assembly | SMART-LLM decompose→coalition→allocate; MetaGPT shared message pool |
| **Unit agents** | partially observable; see own affordances + broadcast feed; emit a *thought* then a command | Dec-POMDP; ReAct interleaved reasoning traces (sparse for decision-making tasks, not dense) |
| **SafetyOfficer** | simulator ground truth as the hard gate; LLM critique for quality | Voyager's self-verification critic, but grounded — strictly stronger than an LLM guess |
| **Reflector** | converts a sparse outcome into one transferable natural-language lesson; capped episodic memory (Ω≈3) | Reflexion verbal reinforcement learning |
| **Skill library** | stores only *validated, non-regressing* plans, keyed by a coarse situation signature; retrieves top-k | Voyager skill library |
| **Conflict-triggered turn-taking** | only units named in a *deterministically detected* conflict get a second turn | DyLAN inference-time agent selection + early stopping, to bound cost |

### The three views given to agents

* **`team_digest`** — shared global state (~250 tokens): score, grid return
  step, known faults + analysis status, unit states, top de-energised loads,
  areas READY to energise, areas still unknown, count of unknown network pixels.
* **`unit_block`** — partially observable per-unit view: own position/state,
  own resources, and **the verbatim JSON of every legal command available to it**.
* **`situation_card`** — mission rules + digest + all unit blocks + the
  deterministic recommendation for every unit. Used by the Dispatcher and by
  the single-agent baselines.

---

## 4. Dense progress potential Φ

Before the transmission grid returns (default step 6) and while the SW block is
unknown, `Δ(weighted restored load)` is **zero for most of the episode**. Ranking
candidates on Δ alone is therefore blind exactly when the decisions matter. We
add (`metrics.py`):
Φ = -3·(remaining repair work)
+15·(#analysed active faults)
+ 1·(#known node/line elements)
+40·(#energizable areas)

and score candidates as `Δweighted + λ·ΔΦ + (critic score)`, where the critic
score is a tie-break only.

---

## 5. Files

| File | Role |
|---|---|
| `affordances.py` | legal menus, greedy policy, the three text views |
| `switching.py` | environment-side greedy switch reconfiguration (no LLM) |
| `patrol.py` | reveal-efficient patrol: greedy max-coverage over unknown *network* pixels |
| `topology.py` | static geometry read from the simulator's own JSON files |
| `protocol.py` | `Message` wire format + deterministic conflict detection |
| `blackboard.py` | append-only shared message pool with per-agent visibility |
| `roles.py` | system prompts + strict output parsers for every role |
| `dialogue.py` | the bounded multi-turn orchestrator (the heart of the system) |
| `memory.py` | trajectory / reflections (Reflexion) / skill library (Voyager) / loop detector |
| `metrics.py` | Φ potential + communication-quality metrics from a transcript |
| `scenarios.py` | randomized fault + unknown-region generation with file backup/restore |
| `agents.py` | single-agent Generator/Verifier/Selector ablation arm |
| `client.py` | in-process HF model, **batched** generation, robust JSON extraction |
| `sim_api.py` | thin wrapper over the DSR V23 HTTP interface |
| `run_experiment.py` | the runner |
| `transcript.py` | JSONL blackboard → readable Markdown transcript |
| `analyze.py` | results.csv + transcripts → the paper tables |
| `trace_log.py` | verbatim prompt/response JSONL logger |
| `scripted.py` | thin wrapper exposing the greedy policy as a policy object |

---

## 6. The multi-turn conversation protocol

### Turn schedule for one simulator time step
ROUND 0 Dispatcher -> ALL digest situation (≤2 sent.) + one intent per unit
ROUND 1 RC1 RC2 PAC1 MPS1-> ALL proposal thought (≤2 sent.) + command + confidence
(optional) -> peer request "RC2, I have fault A; you take B"
ROUND 2 conflicted units -> ALL objection yield or justify [only if conflict]
ROUND 3 Dispatcher -> ALL assignment joint mobile_commands (switches empty)
ROUND 3b SwitchOperator -> ALL switchplan validated toggles + measured gain [no LLM]
ROUND 4 SafetyOfficer -> ALL verdict simulator truth + critique
↳ reject & budget left → back to ROUND 1 with feedback (max_revisions)
↳ still failing → fallback: validated greedy, else no-op
ROUND 5 Environment -> ALL execution score / weighted / locked / energized
Reflector -> ALL reflection one transferable lesson [on failure/loop]

Cost envelope: 1 (dispatcher) + 1 *batched* (units) + ≤1 *batched*
(negotiation) + 1 (critic) + ≤1 (reflector) ≈ **4–5 forward passes per step**,
not 4 sequential per-unit calls. At ~2–4 s/pass for a 7B that is ~15 s/step,
inside LARA's reported sub-20-second decision-latency envelope.

### Per-message JSON contracts

```jsonc
// Dispatcher, round 0
{"situation":"Grid returns at T6; fault A unanalysed at (12,-8); SW network unknown.",
 "intents":{"RC1":"reach fault A and analyse it","RC2":"stage near N13 for fault B",
            "PAC1":"reveal L11/L12 pixels in the SW unknown block",
            "MPS1":"hold; no safe island yet"},
 "risk":"Do not close any switch into A4 before A is repaired."}

// Unit, round 1
{"thought":"A is 8 px away on a level-2 road; arriving costs one locked analysis step, so go now.",
 "command":{"type":"move","waypoints":[[12,-8]]},
 "request":{"to":"RC2","text":"I take A. You keep B so we do not both idle on one pixel."},
 "confidence":0.8}

// Unit, round 2 (objection)
{"thought":"RC2 has repair_speed 8 vs my 5; it should take A and I take the patrol frontier.",
 "command":{"type":"move","waypoints":[[-4,-4]]},"request":null,"confidence":0.6}

// SafetyOfficer, round 4
{"verdict":"accept","risk":"low","score":7,
 "reasons":["no energized load lost","two faults now approached in parallel"],
 "suggestion":"PAC1 should include N11 in its route; it gates area A4."}

// Reflector, round 5
{"lesson":"We wasted two steps because both crews walked to fault A; assign faults by repair_speed at round 0, not after a collision.",
 "tags":["sequencing","repair"]}
 Example rendered transcript (python -m llm_agents.transcript ...)
 ## TIME STEP 3

- **Dispatcher** `digest`
  > Fault A known at (12,-8) but UNANALYSED. SW network still unknown; grid returns T6.
  ```json
  {"intents":{"RC1":"analyse A","RC2":"stage for B","PAC1":"patrol L11/L12","MPS1":"hold"}}
  RC1 proposal
I am already en route with 3 px remaining; continue so I do not reset my path.
{"command":{"type":"continue"},"confidence":0.9}
RC2 proposal
{"command":{"type":"move","waypoints":[[-4,-8]]},"confidence":0.6}
PAC1 proposal → RC2
I will cover L11 and L12 which gate area A4; RC2 should not duplicate my sweep.
{"command":{"type":"move","waypoints":[[-4,-4],[-4,-8],[0,-8]]},"confidence":0.8}
RC2 objection
Yielding the SW sweep to PAC1; I reroute to stage at N13 instead.
{"command":{"type":"move","waypoints":[[0,-16]]}}
SwitchOperator switchplan
baseline |P|*W = 0; no switch toggle improves restored load this step; validate() calls used: 9
Dispatcher assignment
{"mobile_commands":{"RC1":{"type":"continue"},"RC2":{"type":"move","waypoints":[[0,-16]]},"PAC1":{"type":"move","waypoints":[[-4,-4],[-4,-8],[0,-8]]},"MPS1":{"type":"continue"}},"switch_commands":{}}
SafetyOfficer verdict
no energized load lost; patrol covers the two lines gating A4
{"verdict":"accept","risk":"none","score":8,"suggestion":""}
Environment execution
accepted
{"score":0,"weighted":0,"locked":false,"energized":[]}


---

## 7. Ablation ladder

| Method | Isolates |
|---|---|
| `noop` | environment floor |
| `scripted` | greedy heuristic, no switching |
| **`scripted_switch`** | **the real baseline — everything above must beat this** |
| `gen_only` | one LLM, no verification → cost of hallucination |
| `gen_ver` | + simulator hard gate → value of grounded verification |
| `gen_ver_llm` | + LLM critic → value of self-evaluation |
| `ma_nocomm` | role decomposition **without** communication |
| `ma_comm` | + broadcast + negotiation → **the causal test of communication (7 vs 8)** |
| `ma_comm_reflect` | + Reflexion verbal memory across trials |
| `ma_full` | + Voyager skill library |

**The headline number is `score(ma_comm) − score(ma_nocomm)` on matched seeds,
paired t-test.** That is the only clean evidence that *language coordination*
helps, and it is the axis on which this work differs from LARA (single LLM) and
from CTDE-MARL.

Secondary headlines: **lock rate** (fraction of episodes ending with a
permanently-zeroed score) and **decision latency**, not just mean score.

---

## 8. Metrics logged

**Task.** final score, `%scripted_switch`, restoration timesteps, lock rate,
`all_loads_restored` rate, wall-clock/step, LLM calls/step.

**Communication quality** (from the transcript, via `metrics.communication_metrics`):

* `messages_per_step`, `llm_calls_per_step`, `total_response_chars` — cost
* `schema_fail_rate` — can the model even emit the protocol
* `proposal_survival_rate` — fraction of unit proposals surviving assembly
* `n_conflicts` before vs `n_conflicts_after` negotiation — **conflict resolution rate**
* `safety_catches` — how often the deterministic sanitizer had to repair an
  illegal proposal. **This is the hallucination counter**, the analogue of the
  hallucination rates ReAct and Reflexion report as headline numbers.
* `requests_sent`, `objections` — was anyone actually talking
* `verdict_reject_rate`, `revisions` — critic utility
* **deviation value**: `mean(Δ | command ≠ recommendation) − mean(Δ | command = recommendation)`.
  If negative, the LLM is net-harmful and the prompt should push harder toward
  the recommendation.

---

## 9. Suggested sweeps

* **Backbones** (portability): Qwen2.5-7B/14B-Instruct, Llama-3.1-8B-Instruct,
  Mistral-7B-Instruct-v0.3, Qwen2.5-Coder-7B. Expect a wide spread; Voyager
  found generation quality dominant and weaker models collapsing.
* `--k {1,3,5}` × temperature `{0.2,0.5,0.8}` — LARA found `k=3, T=0.2` optimal
  with a sharp drop at `T=0.8`.
* `--max-revisions {0,1,2}` — cost/quality curve of the revise loop.
* `--trials {1..4}` with reflections persisting — the Reflexion learning curve
  (proportion of scenarios solved vs trial number).
* **Personality ablation**: give PAC1 a "curious/explore-first" prompt and RC2 a
  "conservative/never-risk-a-restored-load" prompt (edit `ROLE_BRIEFS` in
  `roles.py`); test whether heterogeneous prompts beat homogeneous ones.
* **Human-on-the-loop**: inject a `K_PATCH` message mid-episode ("the road to
  (12,-8) is blocked; RC2 speed halved") and measure recovery vs a no-patch
  control. This is exactly the capability MILP cannot offer without a cold
  restart, and ReAct showed a human can fix a whole trajectory by editing two
  thoughts.

---

## 10. Troubleshooting order

1. **`scripted_switch` scores ~0.** Stop. The problem is environment wiring,
   not the model. Open `transcript_scripted_switch_ep0_t0.jsonl` and read the
   `switchplan` messages: if every step says "no switch toggle improves restored
   load", the network is never becoming known+fault-free, so look at the patrol
   routes in `trace_*.json`.
2. **Every episode ends `locked=True`.** Something is moving a Supplying MPS.
   Grep the transcript for `supplying_mps_disturbed`.
3. **`schema_fail_rate` > 0.3.** The prompt is too long for the model. Shorten
   `MISSION_BRIEF`, or drop `use_skills`, or raise `max_new_tokens`.
4. **`safety_catches` very high.** The model is ignoring the menus. Raise the
   emphasis on the RECOMMENDED action, or lower `temperature_unit`.
5. **`proposal_survival_rate` ≈ 1.0 and `objections` = 0.** Nobody is actually
   negotiating; `ma_comm` will not beat `ma_nocomm`. Check that
   `use_broadcast=True` and that conflicts are being detected at all.
6. **OOM.** `--load-in-4bit`, or `--max-batch 2`.


# Getting GPU Node
srun --account=torch_pr_111_general --nodes=1 --cpus-per-task=8 \
     --gres=gpu:1 --mem=64G --time=04:00:00 --pty bash

Env Activation:-
/scratch/$USER/dsr_llm/run-dsr.bash /bin/bash
Then:-
source /scratch/sk13001/miniforge3/etc/profile.d/conda.sh
conda activate /scratch/$USER/conda_envs/dsr_llm

export HF_HOME=$SCRATCH/hf_cache        # NOT $HOME — 30K file limit
export MPLCONFIGDIR=$SCRATCH/mpl_cache  # matplotlib defaults to $HOME
export TOKENIZERS_PARALLELISM=false
# export HF_HUB_OFFLINE=1               # once weights are cached

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
nvidia-smi