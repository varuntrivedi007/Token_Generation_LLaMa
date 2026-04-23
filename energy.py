import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import torch

from benchmark import build_prompt, load_model, env_info, Timer


POWER_RE = re.compile(r"(CPU|GPU) Power:\s+(\d+)\s+mW")


def start_powermetrics(sample_ms: int = 100):
    cmd = [
        "powermetrics",
        "--samplers", "cpu_power,gpu_power",
        "-i", str(sample_ms),
        "-f", "text",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        preexec_fn=os.setsid,
    )
    return proc


def parse_power(text: str):
    samples = {"cpu_mW": [], "gpu_mW": []}
    for m in POWER_RE.finditer(text):
        comp, mw = m.group(1), int(m.group(2))
        if comp == "CPU":
            samples["cpu_mW"].append(mw)
        else:
            samples["gpu_mW"].append(mw)
    return samples


@torch.inference_mode()
def workload(model, input_ids, output_tokens, device):
    timer = Timer(device)
    timer.start()
    out = model(input_ids=input_ids.to(model.device), use_cache=True)
    past = out.past_key_values
    tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    for _ in range(output_tokens - 1):
        out = model(input_ids=tok, past_key_values=past, use_cache=True)
        past = out.past_key_values
        tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    return timer.stop()


def run(args):
    if os.geteuid() != 0:
        raise SystemExit("energy.py must be run with sudo for powermetrics access.")

    model, tokenizer = load_model(args.model, "fp16", args.device)
    input_ids = build_prompt(tokenizer, args.prompt_length)

   
    _ = workload(model, input_ids, min(8, args.output_tokens), args.device)

    proc = start_powermetrics(sample_ms=args.sample_ms)
    time.sleep(0.5)  

    durations = []
    for _ in range(args.trials):
        durations.append(workload(model, input_ids, args.output_tokens, args.device))

    time.sleep(0.5)
    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    stdout, _ = proc.communicate(timeout=5)
    samples = parse_power(stdout)

    def mean(xs): return sum(xs) / len(xs) if xs else 0.0

    avg_cpu_mW = mean(samples["cpu_mW"])
    avg_gpu_mW = mean(samples["gpu_mW"])
    avg_total_W = (avg_cpu_mW + avg_gpu_mW) / 1000.0
    total_tokens = args.output_tokens * args.trials
    total_time_s = sum(durations) / 1000.0
    energy_J = avg_total_W * total_time_s
    energy_per_token_mJ = (energy_J / total_tokens) * 1000.0 if total_tokens else 0.0

    out = {
        "env": env_info(args.device),
        "prompt_length": args.prompt_length,
        "output_tokens": args.output_tokens,
        "trials": args.trials,
        "per_trial_ms": durations,
        "avg_cpu_mW": avg_cpu_mW,
        "avg_gpu_mW": avg_gpu_mW,
        "avg_total_W": avg_total_W,
        "total_time_s": total_time_s,
        "total_tokens": total_tokens,
        "energy_J": energy_J,
        "energy_per_token_mJ": energy_per_token_mJ,
        "n_cpu_samples": len(samples["cpu_mW"]),
        "n_gpu_samples": len(samples["gpu_mW"]),
    }

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"results/energy_{args.device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"energy_{args.model}_{args.device}_p{args.prompt_length}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {path}")
    print(f"Energy per token: {energy_per_token_mJ:.2f} mJ  (total {energy_J:.2f} J over {total_tokens} tokens)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["mps", "cpu"], required=True)
    p.add_argument("--model", default="tinyllama")
    p.add_argument("--prompt-length", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--sample-ms", type=int, default=100)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
