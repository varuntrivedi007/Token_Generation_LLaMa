import argparse
import gc
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


GGUF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
GGUF_FILES = {
    "q8": "tinyllama-1.1b-chat-v1.0.Q8_0.gguf",
    "q4": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    "fp16": "tinyllama-1.1b-chat-v1.0.fp16.gguf",
}

DEFAULT_PROMPT_LENGTHS = [32, 64, 128, 256, 512, 1024]
DEFAULT_OUTPUT_TOKENS = 128
DEFAULT_WARMUP = 3
DEFAULT_TRIALS = 10


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


def download_gguf(precision: str) -> str:
    if precision not in GGUF_FILES:
        raise ValueError(f"No GGUF mapping for precision={precision}")
    from huggingface_hub import hf_hub_download
    cache_dir = os.environ.get("GGUF_CACHE", str(Path.home() / ".cache" / "gguf_models"))
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    print(f"[download] {GGUF_REPO}/{GGUF_FILES[precision]}")
    path = hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILES[precision],
        cache_dir=cache_dir,
    )
    return path


def load_llama(gguf_path: str, device: str, n_ctx: int):
    
    from llama_cpp import Llama
    n_gpu_layers = -1 if device in ("mps", "cuda") else 0
    kwargs = dict(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        logits_all=False,
        verbose=False,
        seed=0,
    )
    return Llama(**kwargs)


def build_prompt_tokens(llm, target_tokens: int) -> List[int]:
    
    filler = (
        "The quick brown fox jumps over the lazy dog. "
        "Language models generate one token at a time. "
    )
    text = filler * (target_tokens // 8 + 8)
    toks = llm.tokenize(text.encode("utf-8"), add_bos=True)
    if len(toks) < target_tokens:
        # Repeat tail tokens (not BOS) to reach length
        tail = toks[1:] if len(toks) > 1 else toks
        while len(toks) < target_tokens:
            toks = toks + tail
    return toks[:target_tokens]


def run_trial(llm, input_tokens: List[int], output_tokens: int) -> TrialResult:
    llm.reset()

    per_token_ms: List[float] = []
    total_start = time.perf_counter_ns()

    
    t0 = time.perf_counter_ns()
    llm.eval(input_tokens)
    next_tok = int(llm.sample(top_k=1, top_p=1.0, temp=0.0))
    ttft_ms = (time.perf_counter_ns() - t0) / 1e6
    per_token_ms.append(ttft_ms)

    for _ in range(output_tokens - 1):
        t = time.perf_counter_ns()
        llm.eval([next_tok])
        next_tok = int(llm.sample(top_k=1, top_p=1.0, temp=0.0))
        per_token_ms.append((time.perf_counter_ns() - t) / 1e6)

    total_ms = (time.perf_counter_ns() - total_start) / 1e6
    throughput = output_tokens / (total_ms / 1000.0) if total_ms > 0 else 0.0

    return TrialResult(
        ttft_ms=ttft_ms,
        per_token_ms=per_token_ms,
        total_ms=total_ms,
        output_tokens=output_tokens,
        throughput_tok_per_s=throughput,
    )


def env_info(device: str) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "device": device,
        "cpu": platform.processor(),
        "backend": "llama.cpp",
    }
    try:
        import llama_cpp
        info["llama_cpp_version"] = getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        pass
    return info


def default_output_dir(device: str, platform_tag: Optional[str]) -> Path:
    if platform_tag:
        return Path(f"results/{platform_tag}")
    mapping = {
        "mps": "results/m4pro_mps",
        "cpu": "results/m4pro_cpu",
        "cuda": "results/colab_t4",
    }
    return Path(mapping.get(device, f"results/{device}"))


def run_config(args) -> dict:
    device = args.device
    print(f"[gguf] precision={args.precision} device={device}")
    gguf_path = args.gguf_path or download_gguf(args.precision)
    print(f"[gguf] loaded: {gguf_path}")

    max_plen = max(args.prompt_lengths)
    
    n_ctx = max(2048, max_plen + args.output_tokens + 64)
    llm = load_llama(gguf_path, device, n_ctx=n_ctx)

    results_per_plen = {}

    for plen in args.prompt_lengths:
        print(f"\n[config] prompt_length={plen} output_tokens={args.output_tokens}")
        input_tokens = build_prompt_tokens(llm, plen)
        actual_plen = len(input_tokens)

        for w in range(args.warmup):
            print(f"  [warmup {w + 1}/{args.warmup}]", end="", flush=True)
            _ = run_trial(llm, input_tokens, min(16, args.output_tokens))
            print(" done")

        trials: List[TrialResult] = []
        for t in range(args.trials):
            r = run_trial(llm, input_tokens, args.output_tokens)
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

    return {
        "env": env_info(device),
        "model": args.model,
        "model_id": f"{GGUF_REPO}/{GGUF_FILES[args.precision]}",
        "backend": "llama.cpp",
        "precision": args.precision,
        "device": device,
        "platform_tag": args.platform_tag,
        "warmup": args.warmup,
        "trials": args.trials,
        "output_tokens": args.output_tokens,
        "results": results_per_plen,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["mps", "cuda", "cpu"], required=True,
                   help="Backend tag. Controls n_gpu_layers and results grouping.")
    p.add_argument("--platform-tag", type=str, default=None,
                   help="Platform dir name (e.g. m4pro_mps, windows_cpu). Defaults per device.")
    p.add_argument("--model", type=str, default="tinyllama")
    p.add_argument("--precision", choices=["q8", "q4", "fp16"], required=True)
    p.add_argument("--gguf-path", type=str, default=None,
                   help="Local GGUF file path. If unset, downloads from HF.")
    p.add_argument("--prompt-lengths", type=int, nargs="+",
                   default=DEFAULT_PROMPT_LENGTHS)
    p.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    data = run_config(args)

    out_dir = (Path(args.output_dir) if args.output_dir
               else default_output_dir(args.device, args.platform_tag))
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"_{args.tag}" if args.tag else ""
    fname = f"{args.model}_{args.precision}_{args.device}_gguf{tag}.json"
    out_path = out_dir / fname
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
