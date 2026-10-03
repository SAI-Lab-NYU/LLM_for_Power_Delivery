"""In-process HuggingFace LLM client with BATCHED generation.

Key properties:
  * complete_batch(): one forward pass for many independent prompts. The unit
    agents in a dialogue round are independent, so this is a ~Nx latency win
    over a sequential per-agent loop.
  * HARD DEVICE GUARD: device_map="auto" silently falls back to CPU when CUDA
    is invisible (login node, srun without --gres=gpu, singularity without
    --nv). A 7B on CPU is ~50-200x slower and makes decision-latency numbers
    meaningless. We therefore REFUSE to load on CPU unless DSR_ALLOW_CPU=1.
  * torch_dtype / dtype kwarg fallback so it works on old and new transformers.
  * left padding + pad token, which batched decoder-only generation requires.
  * seeding for reproducibility; optional 4-bit loading for tight GPUs.
  * extract_json(): balanced-brace scanner, robust to NESTED objects.
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


class CPUFallbackError(RuntimeError):
    """Raised when no accelerator is available and CPU was not explicitly allowed."""


@dataclass
class LLMConfig:
    model: str = "Qwen/Qwen2.5-7B-Instruct"
    temperature: float = 0.3
    max_new_tokens: int = 384          # 256 was too small once thoughts exist
    dtype: str = "bfloat16"
    seed: int = 0
    load_in_4bit: bool = False
    max_batch: int = 8                 # split larger batches to bound memory
    allow_cpu: bool = False            # or set DSR_ALLOW_CPU=1


class LLMClient:
    """Loads the model once, reuses it for every generate call."""

    def __init__(self, config: LLMConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self._check_device(config, torch)

        torch.manual_seed(int(config.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(config.seed))

        print(f"[llm] loading {config.model} ...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # REQUIRED for batched generation on decoder-only models
        self.tokenizer.padding_side = "left"

        self.model = self._load_model(config, torch, AutoModelForCausalLM)
        self.model.eval()

        self._report_placement(torch)

        self.last_latency_s = 0.0
        self.total_calls = 0
        self.total_latency_s = 0.0
        self.total_prompt_chars = 0
        self.total_response_chars = 0

    # ------------------------------------------------------------------ #
    @staticmethod
    def _check_device(config: LLMConfig, torch) -> None:
        """Fail fast and loudly rather than silently crawling on CPU."""
        allow_cpu = bool(config.allow_cpu) or \
            os.environ.get("DSR_ALLOW_CPU", "").strip() in ("1", "true", "yes")

        cuda = torch.cuda.is_available()
        mps = bool(getattr(getattr(torch, "backends", None), "mps", None)
                   and torch.backends.mps.is_available())

        if cuda or mps:
            return
        if allow_cpu:
            print("[llm] WARNING: no accelerator found; running on CPU because "
                  "DSR_ALLOW_CPU is set. Expect ~50-200x slower generation and "
                  "meaningless decision-latency numbers.", flush=True)
            return

        raise CPUFallbackError(
            "No CUDA/MPS device is visible, so this model would load on CPU.\n"
            "  A 7B model on CPU is ~50-200x slower: a single episode can take\n"
            "  hours instead of seconds, and any latency metric you report is\n"
            "  invalid.\n\n"
            "  Diagnose:\n"
            "    nvidia-smi\n"
            "    python -c \"import torch; print(torch.cuda.is_available())\"\n"
            f"    CUDA_VISIBLE_DEVICES=[{os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')}]\n\n"
            "  Common causes:\n"
            "    * you are on a login node -> request a GPU node\n"
            "    * srun/sbatch is missing  --gres=gpu:1\n"
            "    * singularity is missing  --nv\n"
            "    * CUDA_VISIBLE_DEVICES is set to an empty string\n\n"
            "  To run on CPU anyway (smoke tests only):  DSR_ALLOW_CPU=1 ...")

    @staticmethod
    def _load_model(config: LLMConfig, torch, AutoModelForCausalLM):
        kwargs = dict(device_map="auto", trust_remote_code=True)
        if config.load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, config.dtype),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True)
            return AutoModelForCausalLM.from_pretrained(config.model, **kwargs)

        dt = getattr(torch, config.dtype)
        try:                                            # transformers >= 4.56
            return AutoModelForCausalLM.from_pretrained(
                config.model, dtype=dt, **kwargs)
        except TypeError:                               # older: torch_dtype
            return AutoModelForCausalLM.from_pretrained(
                config.model, torch_dtype=dt, **kwargs)

    def _report_placement(self, torch) -> None:
        dev = str(getattr(self.model, "device", "unknown"))
        print(f"[llm] loaded on {dev}", flush=True)

        # device_map="auto" can shard: surface any CPU/disk offload explicitly.
        dmap = getattr(self.model, "hf_device_map", None)
        if isinstance(dmap, dict):
            placements = sorted({str(v) for v in dmap.values()})
            if len(placements) > 1:
                print(f"[llm] device map spans: {placements}", flush=True)
            offloaded = [k for k, v in dmap.items()
                         if str(v) in ("cpu", "disk")]
            if offloaded:
                print(f"[llm] WARNING: {len(offloaded)} module(s) offloaded to "
                      f"cpu/disk. Generation will be much slower. Consider "
                      f"--load-in-4bit or a larger-memory GPU.", flush=True)

        if dev.startswith("cpu"):
            print("[llm] WARNING: model is on CPU despite the device check.",
                  flush=True)
        elif torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info()
                print(f"[llm] GPU memory: {(total - free) / 2**30:.1f} GiB used "
                      f"/ {total / 2**30:.1f} GiB total", flush=True)
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    def _prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------ #
    def complete(self, system: str, user: str,
                 temperature: Optional[float] = None) -> str:
        text, _ = self.complete_timed(system, user, temperature)
        return text

    def complete_timed(self, system: str, user: str,
                       temperature: Optional[float] = None) -> Tuple[str, float]:
        out = self.complete_batch([(system, user)], temperature=temperature)
        return out[0]

    def complete_batch(self,
                       pairs: Sequence[Tuple[str, str]],
                       temperature: Optional[float] = None,
                       max_new_tokens: Optional[int] = None
                       ) -> List[Tuple[str, float]]:
        """Generate one completion per (system, user) pair in ONE forward pass.

        Returns [(text, per_item_latency_s), ...] in the same order as `pairs`.
        """
        import torch

        if not pairs:
            return []

        temp = self.config.temperature if temperature is None else float(temperature)
        mnt = int(max_new_tokens or self.config.max_new_tokens)

        results: List[Tuple[str, float]] = []
        step = max(1, int(self.config.max_batch))
        for start in range(0, len(pairs), step):
            chunk = list(pairs[start:start + step])
            prompts = [self._prompt(s, u) for s, u in chunk]
            enc = self.tokenizer(prompts, return_tensors="pt",
                                 padding=True).to(self.model.device)

            t0 = time.time()
            with torch.no_grad():
                out = self.model.generate(
                    **enc,
                    max_new_tokens=mnt,
                    do_sample=temp > 0.01,
                    temperature=max(temp, 0.01),
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id)
            latency = time.time() - t0

            n_in = enc["input_ids"].shape[1]
            texts = [self.tokenizer.decode(seq[n_in:], skip_special_tokens=True)
                     for seq in out]

            per = latency / max(len(texts), 1)
            self.last_latency_s = per
            self.total_calls += len(texts)
            self.total_latency_s += latency
            self.total_prompt_chars += sum(len(p) for p in prompts)
            self.total_response_chars += sum(len(t) for t in texts)
            results.extend((t, per) for t in texts)

        return results

    # ------------------------------------------------------------------ #
    def stats(self) -> dict:
        mean = (self.total_latency_s / self.total_calls) if self.total_calls else 0.0
        return {"calls": self.total_calls,
                "total_latency_s": round(self.total_latency_s, 1),
                "mean_latency_s": round(mean, 2),
                "device": str(getattr(self.model, "device", "unknown")),
                "prompt_chars": self.total_prompt_chars,
                "response_chars": self.total_response_chars}


# ---------------------------------------------------------------------- #
def extract_json(text: str) -> Optional[dict]:
    """Recover the first complete JSON OBJECT from an LLM reply.

    Uses a balanced-brace scan that is string-literal and escape aware, so it
    handles nested objects (our command schema is nested), prose before/after,
    and markdown fences.
    """
    if not text:
        return None

    candidates = [text]
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        candidates.insert(0, fence.group(1))

    for s in candidates:
        for start in (i for i, ch in enumerate(s) if ch == "{"):
            depth, in_str, esc = 0, False, False
            for i in range(start, len(s)):
                ch = s[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        blob = s[start:i + 1]
                        try:
                            parsed = json.loads(blob)
                        except json.JSONDecodeError:
                            try:                       # tolerate trailing commas
                                parsed = json.loads(
                                    re.sub(r",\s*([}\]])", r"\1", blob))
                            except json.JSONDecodeError:
                                break
                        if isinstance(parsed, dict):
                            return parsed
                        break
    return None