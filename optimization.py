"""
Phase 4: INT8 KV-cache quantization.

Wraps a LLaMA model so that past K/V tensors are stored as int8 (per-head,
per-token symmetric quantization) and dequantized to fp16 on read. Compares
latency, memory, and perplexity against the unquantized baseline.
"""

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from benchmark import MODEL_IDS, Timer, build_prompt, load_model, summarize, env_info


def _extract_kv_pairs(cache):
    """Extract (k, v) pairs from any DynamicCache variant or legacy tuple."""
    
    if hasattr(cache, "layers") and len(getattr(cache, "layers", [])) > 0 \
       and hasattr(cache.layers[0], "keys"):
        return [(L.keys, L.values) for L in cache.layers]
    # Mid-era: .key_cache / .value_cache lists
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    # Legacy: to_legacy_cache() returns tuple of (k,v) per layer
    if hasattr(cache, "to_legacy_cache"):
        try:
            legacy = cache.to_legacy_cache()
            return [(k, v) for (k, v) in legacy]
        except Exception:
            pass
    # Fallback: plain tuple of tuples
    return [(k, v) for (k, v) in cache]


def _rebuild_cache_from_pairs(pairs):
    """Rebuild a DynamicCache from (k, v) pairs (one per layer)."""
    cache = DynamicCache()
    for i, (k, v) in enumerate(pairs):
        cache.update(k, v, i)
    return cache


class QuantizedKVCache:
    """
    Per-layer INT8 symmetric-quantized KV cache.

    Stores K and V as int8 tensors plus a per-(layer, head, token) fp16 scale.
    dequantize() rematerializes fp16 tensors for attention compute.
    """

    def __init__(self, num_layers: int):
        self.k_int8: List[torch.Tensor] = [None] * num_layers
        self.v_int8: List[torch.Tensor] = [None] * num_layers
        self.k_scale: List[torch.Tensor] = [None] * num_layers
        self.v_scale: List[torch.Tensor] = [None] * num_layers

    @staticmethod
    def _quant(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B, H, T, D]. Per (B, H, T) scale.
        absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = absmax / 127.0
        q = torch.clamp(torch.round(x / scale), -127, 127).to(torch.int8)
        return q, scale.to(torch.float16)

    @staticmethod
    def _dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q.to(torch.float16) * scale

    def append(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        kq, ks = self._quant(k)
        vq, vs = self._quant(v)
        if self.k_int8[layer_idx] is None:
            self.k_int8[layer_idx] = kq
            self.v_int8[layer_idx] = vq
            self.k_scale[layer_idx] = ks
            self.v_scale[layer_idx] = vs
        else:
            self.k_int8[layer_idx] = torch.cat([self.k_int8[layer_idx], kq], dim=-2)
            self.v_int8[layer_idx] = torch.cat([self.v_int8[layer_idx], vq], dim=-2)
            self.k_scale[layer_idx] = torch.cat([self.k_scale[layer_idx], ks], dim=-2)
            self.v_scale[layer_idx] = torch.cat([self.v_scale[layer_idx], vs], dim=-2)

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self._dequant(self.k_int8[layer_idx], self.k_scale[layer_idx]),
            self._dequant(self.v_int8[layer_idx], self.v_scale[layer_idx]),
        )

    def bytes_used(self) -> int:
        total = 0
        for t in self.k_int8 + self.v_int8:
            if t is not None:
                total += t.numel()  # 1 byte each
        for t in self.k_scale + self.v_scale:
            if t is not None:
                total += t.numel() * 2  # fp16
        return total


# ---------------------------------------------------------------------------
# Decode loop with quantized KV cache
#
# Hugging Face's LLaMA stores past_key_values as a tuple of (k, v) per layer.
# We intercept after each forward to re-quantize the growing cache, then
# replace the model's cache with a dequantized version on the next step.
# ---------------------------------------------------------------------------

@torch.inference_mode()
def generate_quantized_kv(model, tokenizer, input_ids, output_tokens, device) -> dict:
    input_ids = input_ids.to(model.device)

    timer = Timer(device)
    timer.start()

    # Prefill
    out = model(input_ids=input_ids, use_cache=True)
    pairs = _extract_kv_pairs(out.past_key_values)
    num_layers = len(pairs)
    qcache = QuantizedKVCache(num_layers)
    fp_past = []
    for i, (k, v) in enumerate(pairs):
        qcache.append(i, k, v)
        dk, dv = qcache.get(i)
        fp_past.append((dk, dv))
    past = _rebuild_cache_from_pairs(fp_past)

    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    for _ in range(output_tokens - 1):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        new_pairs = _extract_kv_pairs(out.past_key_values)
        qcache = QuantizedKVCache(num_layers)
        fp_past = []
        for i, (k, v) in enumerate(new_pairs):
            qcache.append(i, k, v)
            dk, dv = qcache.get(i)
            fp_past.append((dk, dv))
        past = _rebuild_cache_from_pairs(fp_past)
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    total_ms = timer.stop()
    return {
        "total_ms": total_ms,
        "kv_bytes_int8": qcache.bytes_used(),
        "throughput_tok_per_s": output_tokens / (total_ms / 1000.0),
    }


@torch.inference_mode()
def generate_baseline(model, tokenizer, input_ids, output_tokens, device) -> dict:
    input_ids = input_ids.to(model.device)
    timer = Timer(device)
    timer.start()
    out = model(input_ids=input_ids, use_cache=True)
    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    for _ in range(output_tokens - 1):
        out = model(input_ids=next_token, past_key_values=past, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)
    total_ms = timer.stop()

    kv_bytes = 0
    for k, v in _extract_kv_pairs(past):
        kv_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
    return {
        "total_ms": total_ms,
        "kv_bytes_fp16": kv_bytes,
        "throughput_tok_per_s": output_tokens / (total_ms / 1000.0),
    }



@torch.inference_mode()
def perplexity(model, tokenizer, text: str, device: str, use_qkv: bool = False) -> float:
    ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
    if not use_qkv:
        out = model(input_ids=ids, labels=ids)
        return math.exp(out.loss.item())
    # Quantized path: run token-by-token with requantized KV
    losses = []
    past = None
    for t in range(1, ids.shape[1]):
        step_input = ids[:, :1] if past is None else ids[:, t - 1:t]
        if past is None:
            step_input = ids[:, :t]
        out = model(input_ids=step_input, past_key_values=past, use_cache=True)
        pairs = _extract_kv_pairs(out.past_key_values)
        num_layers = len(pairs)
        qcache = QuantizedKVCache(num_layers)
        fp_past = []
        for i, (k, v) in enumerate(pairs):
            qcache.append(i, k, v)
            dk, dv = qcache.get(i)
            fp_past.append((dk, dv))
        past = _rebuild_cache_from_pairs(fp_past)
        logits = out.logits[:, -1, :]
        target = ids[:, t]
        loss = F.cross_entropy(logits, target)
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))



def run(args):
    device = args.device
    model, tokenizer = load_model(args.model, "fp16", device)

    results = {"baseline": {}, "quantized": {}, "perplexity": {}}
    eval_text = (
        "Attention is all you need. Transformers have revolutionized sequence "
        "modeling by replacing recurrence with self-attention over token "
        "representations, enabling highly parallel training and scalable inference."
    )

    for plen in args.prompt_lengths:
        print(f"\n=== prompt_length={plen} ===")
        input_ids = build_prompt(tokenizer, plen)

        
        _ = generate_baseline(model, tokenizer, input_ids, min(8, args.output_tokens), device)

        base_trials = []
        for i in range(args.trials):
            r = generate_baseline(model, tokenizer, input_ids, args.output_tokens, device)
            base_trials.append(r)
            print(f"[baseline {i+1}] total={r['total_ms']:.1f}ms kv={r['kv_bytes_fp16']/1e6:.2f}MB")

        q_trials = []
        for i in range(args.trials):
            r = generate_quantized_kv(model, tokenizer, input_ids, args.output_tokens, device)
            q_trials.append(r)
            print(f"[int8-kv {i+1}] total={r['total_ms']:.1f}ms kv={r['kv_bytes_int8']/1e6:.2f}MB")

        results["baseline"][str(plen)] = {
            "total_ms": summarize([t["total_ms"] for t in base_trials]),
            "kv_bytes_fp16_mean": sum(t["kv_bytes_fp16"] for t in base_trials) / len(base_trials),
            "throughput": summarize([t["throughput_tok_per_s"] for t in base_trials]),
        }
        results["quantized"][str(plen)] = {
            "total_ms": summarize([t["total_ms"] for t in q_trials]),
            "kv_bytes_int8_mean": sum(t["kv_bytes_int8"] for t in q_trials) / len(q_trials),
            "throughput": summarize([t["throughput_tok_per_s"] for t in q_trials]),
        }

   
    try:
        ppl_base = perplexity(model, tokenizer, eval_text, device, use_qkv=False)
        ppl_q = perplexity(model, tokenizer, eval_text, device, use_qkv=True)
        results["perplexity"] = {"baseline": ppl_base, "int8_kv": ppl_q}
        print(f"\nPerplexity  baseline={ppl_base:.3f}  int8_kv={ppl_q:.3f}")
    except Exception as e:
        results["perplexity"] = {"error": str(e)}

    out = {
        "env": env_info(device),
        "model": args.model,
        "device": device,
        "output_tokens": args.output_tokens,
        "results": results,
    }

    out_dir = Path(args.output_dir) if args.output_dir else Path(f"results/optimization_{device}")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"kvq_{args.model}_{device}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[saved] {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", choices=["mps", "cuda", "cpu"], required=True)
    p.add_argument("--model", choices=list(MODEL_IDS.keys()), default="tinyllama")
    p.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 1024])
    p.add_argument("--output-tokens", type=int, default=64)
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
