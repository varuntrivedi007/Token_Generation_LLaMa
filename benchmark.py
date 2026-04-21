"""
Token-generation latency benchmark harness.

Works across MPS (Apple Silicon), CUDA (NVIDIA), and CPU backends.
Measures TTFT, per-token latency, end-to-end time, throughput with proper
device synchronization, warmup, and IQR-based outlier filtering.
"""

import argparse
import gc
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_IDS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "openllama3b": "openlm-research/open_llama_3b",
}

DEFAULT_PROMPT_LENGTHS = [32, 64, 128, 256, 512, 1024]
DEFAULT_OUTPUT_TOKENS = 128
DEFAULT_WARMUP = 3
DEFAULT_TRIALS = 10


class Timer:
    """Device-aware wall-clock timer returning milliseconds."""

    def __init__(self, device: str):
        self.device = device
        self._start = None
        self._end = None
        self._cuda_start = None
        self._cuda_end = None

    def sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def start(self):
        if self.device == "cuda":
            self._cuda_start = torch.cuda.Event(enable_timing=True)
            self._cuda_end = torch.cuda.Event(enable_timing=True)
            self._cuda_start.record()
        else:
            self.sync()
            self._start = time.perf_counter_ns()

    def stop(self) -> float:
        if self.device == "cuda":
            self._cuda_end.record()
            torch.cuda.synchronize()
            return self._cuda_start.elapsed_time(self._cuda_end)  # ms
        self.sync()
        self._end = time.perf_counter_ns()
        return (self._end - self._start) / 1e6  # ms



def iqr_filter(values: List[float]) -> List[float]:
    if len(values) < 4:
        return list(values)
    s = sorted(values)
    q1 = s[len(s) // 4]
    q3 = s[(3 * len(s)) // 4]
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [v for v in values if lo <= v <= hi]


def summarize(values: List[float]) -> dict:
    filtered = iqr_filter(values)
    return {
        "raw": values,
        "filtered": filtered,
        "n_raw": len(values),
        "n_filtered": len(filtered),
        "mean": statistics.mean(filtered) if filtered else None,
        "median": statistics.median(filtered) if filtered else None,
        "stdev": statistics.stdev(filtered) if len(filtered) > 1 else 0.0,
        "min": min(filtered) if filtered else None,
        "max": max(filtered) if filtered else None,
    }



@dataclass
class TrialResult:
    ttft_ms: float
    per_token_ms: List[float]
    total_ms: float
    output_tokens: int
    throughput_tok_per_s: float



def build_prompt(tokenizer, target_tokens: int) -> torch.Tensor:
    """Build a prompt with exactly target_tokens tokens (approx)."""
    # Use a deterministic filler — repeat a short sentence then truncate.
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Language models generate one token at a time. "
    )
    text = filler * (target_tokens // 8 + 4)
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    if ids.shape[0] < target_tokens:
        # Pad by repeating
        reps = target_tokens // ids.shape[0] + 1
        ids = ids.repeat(reps)
    ids = ids[:target_tokens].unsqueeze(0)
    return ids



@torch.inference_mode()
def run_trial(
    model,
    input_ids: torch.Tensor,
    output_tokens: int,
    device: str,
    eos_token_id: Optional[int] = None,
) -> TrialResult:
    input_ids = input_ids.to(model.device)
    per_token_ms: List[float] = []
    timer = Timer(device)

    # Total timer
    total_timer = Timer(device)
    total_timer.start()


    ttft_timer = Timer(device)
    ttft_timer.start()
    out = model(input_ids=input_ids, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    ttft_ms = ttft_timer.stop()
    per_token_ms.append(ttft_ms)  # first token = TTFT


    for _ in range(output_tokens - 1):
        step_timer = Timer(device)
        step_timer.start()
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        per_token_ms.append(step_timer.stop())

    total_ms = total_timer.stop()
    throughput = output_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return TrialResult(
        ttft_ms=ttft_ms,
        per_token_ms=per_token_ms,
        total_ms=total_ms,
        output_tokens=output_tokens,
        throughput_tok_per_s=throughput,
    )



def load_model(model_key: str, precision: str, device: str):
    model_id = MODEL_IDS[model_key]

    dtype = torch.float16
    bnb_config = None

    if precision == "fp16":
        dtype = torch.float16
    elif precision == "fp32":
        dtype = torch.float32
    elif precision == "q8":
        if device != "cuda":
            raise SystemExit("q8 (bitsandbytes) requires CUDA device.")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "q4":
        if device != "cuda":
            raise SystemExit("q4 (bitsandbytes) requires CUDA device.")
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        raise ValueError(f"Unknown precision: {precision}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if bnb_config is None:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
        model = model.to(device)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", quantization_config=bnb_config
        )

    model.eval()
    return model, tokenizer


def env_info(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": device,
        "cpu": platform.processor(),
    }
    if device == "cuda" and torch.cuda.is_available():
        info["cuda_device"] = torch.cuda.get_device_name(0)
        info["cuda_capability"] = list(torch.cuda.get_device_capability(0))
    return info


def run_config(args) -> dict:
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS not available on this machine.")
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA not available on this machine.")

    print(f"[load] model={args.model} precision={args.precision} device={device}")
    model, tokenizer = load_model(args.model, args.precision, device)

    results_per_plen = {}

    for plen in args.prompt_lengths:
        print(f"\n[config] prompt_length={plen} output_tokens={args.output_tokens}")
        input_ids = build_prompt(tokenizer, plen)
        actual_plen = int(input_ids.shape[1])

        
        for w in range(args.warmup):
            print(f"  [warmup {w + 1}/{args.warmup}]", end="", flush=True)
            _ = run_trial(model, input_ids, min(16, args.output_tokens), device)
            print(" done")

        
        trials: List[TrialResult] = []
        for t in range(args.trials):
            r = run_trial(model, input_ids, args.output_tokens, device)
            trials.append(r)
            print(
                f"  [trial {t + 1}/{args.trials}] "
                f"ttft={r.ttft_ms:.2f}ms "
                f"throughput={r.throughput_tok_per_s:.2f} tok/s "
                f"total={r.total_ms:.2f}ms"
            )

        ttft_vals = [t.ttft_ms for t in trials]
        total_vals = [t.total_ms for t in trials]
        thpt_vals = [t.throughput_tok_per_s for t in trials]
        # Per-token latency: exclude first (TTFT) and compute median per trial
        decode_vals = []
        for t in trials:
            if len(t.per_token_ms) > 1:
                decode_vals.extend(t.per_token_ms[1:])

        results_per_plen[str(plen)] = {
            "actual_prompt_tokens": actual_plen,
            "output_tokens": args.output_tokens,
            "ttft_ms": summarize(ttft_vals),
            "total_ms": summarize(total_vals),
            "throughput_tok_per_s": summarize(thpt_vals),
            "decode_per_token_ms": summarize(decode_vals),
            "raw_trials": [asdict(t) for t in trials],
        }

        
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()

    return {
        "env": env_info(device),
        "model": args.model,
        "model_id": MODEL_IDS[args.model],
        "precision": args.precision,
        "device": device,
        "warmup": args.warmup,
        "trials": args.trials,
        "output_tokens": args.output_tokens,
        "results": results_per_plen,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["mps", "cuda", "cpu"], required=True)
    p.add_argument("--model", choices=list(MODEL_IDS.keys()), default="tinyllama")
    p.add_argument("--precision", choices=["fp16", "fp32", "q8", "q4"], default="fp16")
    p.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_PROMPT_LENGTHS,
    )
    p.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory (default: results/<device-tag>/).",
    )
    p.add_argument("--tag", type=str, default=None, help="Optional run tag.")
    return p.parse_args()


def default_output_dir(device: str) -> Path:
    mapping = {
        "mps": "results/m4pro_mps",
        "cpu": "results/m4pro_cpu",
        "cuda": "results/colab_t4",
    }
    return Path(mapping.get(device, f"results/{device}"))


def main():
    args = parse_args()
    data = run_config(args)

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.device)
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    fname = f"{args.model}_{args.precision}_{args.device}{tag}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
