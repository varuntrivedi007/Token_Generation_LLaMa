import argparse
import gc
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmark import (
    MODEL_IDS,
    Timer,
    build_prompt,
    load_model,
    summarize,
    env_info,
)


COMPONENT_PATTERNS = [
    (re.compile(r"embed_tokens$|embeddings$"), "embedding"),
    (re.compile(r"\.q_proj$|\.k_proj$|\.v_proj$"), "qkv_projection"),
    (re.compile(r"\.o_proj$"), "attn_output_proj"),
    (re.compile(r"self_attn$"), "attention_full"),
    (re.compile(r"\.gate_proj$|\.up_proj$|\.down_proj$"), "mlp_projections"),
    (re.compile(r"\.mlp$"), "mlp_full"),
    (re.compile(r"input_layernorm$|post_attention_layernorm$|\.norm$"), "rmsnorm"),
    (re.compile(r"lm_head$"), "lm_head"),
]


def classify(name: str) -> str:
    for pat, bucket in COMPONENT_PATTERNS:
        if pat.search(name):
            return bucket
    return "other"


class HookTimer:
    
    def __init__(self, model, device: str):
        self.model = model
        self.device = device
        self.pre_times: Dict[str, float] = {}
        self.totals: Dict[str, float] = defaultdict(float)
        self.counts: Dict[str, int] = defaultdict(int)
        self.handles = []

    def _sync(self):
        if self.device == "mps":
            torch.mps.synchronize()
        elif self.device == "cuda":
            torch.cuda.synchronize()

    def _pre_hook(self, name):
        def hook(_module, _inputs):
            self._sync()
            self.pre_times[name] = time.perf_counter_ns()
        return hook

    def _post_hook(self, name):
        def hook(_module, _inputs, _output):
            self._sync()
            start = self.pre_times.pop(name, None)
            if start is None:
                return
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            bucket = classify(name)
            self.totals[bucket] += elapsed_ms
            self.counts[bucket] += 1
        return hook

    def attach(self):
        
        for name, module in self.model.named_modules():
            if name == "":
                continue
            self.handles.append(module.register_forward_pre_hook(self._pre_hook(name)))
            self.handles.append(module.register_forward_hook(self._post_hook(name)))

    def detach(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def reset(self):
        self.pre_times.clear()
        self.totals.clear()
        self.counts.clear()


@torch.inference_mode()
def decompose_trial(model, input_ids, output_tokens, device, hooktimer):
    input_ids = input_ids.to(model.device)
    hooktimer.reset()

    end_to_end = Timer(device)
    end_to_end.start()

    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    for _ in range(output_tokens - 1):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    total_ms = end_to_end.stop()
    component_totals = dict(hooktimer.totals)
    hooked_sum = sum(component_totals.values())
    component_totals["framework_overhead"] = max(total_ms - hooked_sum, 0.0)
    return total_ms, component_totals


def run(args):
    device = args.device
    model, tokenizer = load_model(args.model, args.precision, device)

    input_ids = build_prompt(tokenizer, args.prompt_length)

    hooktimer = HookTimer(model, device)
    hooktimer.attach()

    
    for _ in range(args.warmup):
        _ = decompose_trial(model, input_ids, min(8, args.output_tokens), device, hooktimer)

    trials: List[dict] = []
    for i in range(args.trials):
        total_ms, comps = decompose_trial(
            model, input_ids, args.output_tokens, device, hooktimer
        )
        trials.append({"total_ms": total_ms, "components_ms": comps})
        print(f"[trial {i+1}/{args.trials}] total={total_ms:.1f}ms  "
              + "  ".join(f"{k}={v:.1f}" for k, v in comps.items()))

    
    keys = set()
    for t in trials:
        keys.update(t["components_ms"].keys())
    agg = {}
    for k in keys:
        vals = [t["components_ms"].get(k, 0.0) for t in trials]
        agg[k] = summarize(vals)
    total_agg = summarize([t["total_ms"] for t in trials])

    out = {
        "env": env_info(device),
        "model": args.model,
        "precision": args.precision,
        "device": device,
        "prompt_length": args.prompt_length,
        "output_tokens": args.output_tokens,
        "trials": trials,
        "component_summary_ms": agg,
        "total_summary_ms": total_agg,
    }

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"results/decomposition_{device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"decomp_{args.model}_{args.precision}_{device}_p{args.prompt_length}.json"
    path = out_dir / fname
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {path}")

    hooktimer.detach()
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["mps", "cuda", "cpu"], required=True)
    p.add_argument("--model", choices=list(MODEL_IDS.keys()), default="tinyllama")
    p.add_argument("--precision", choices=["fp16", "fp32", "q8", "q4"], default="fp16")
    p.add_argument("--prompt-length", type=int, default=128)
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
