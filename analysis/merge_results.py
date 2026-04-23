import json
from pathlib import Path
import pandas as pd


def load_all(results_root: str = "results") -> pd.DataFrame:
    rows = []
    root = Path(results_root)
    for path in root.rglob("*.json"):
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        if "results" not in data or not isinstance(data["results"], dict):
            continue
        
        sample = next(iter(data["results"].values()), {})
        if "ttft_ms" not in sample:
            continue
        backend = data.get("backend", "pytorch")
        for plen, r in data["results"].items():
            rows.append({
                "platform_dir": path.parent.name,
                "platform": path.parent.name,
                "file": path.name,
                "device": data.get("device"),
                "backend": backend,
                "model": data.get("model"),
                "precision": data.get("precision"),
                "prompt_length": int(plen),
                "ttft_ms_median": r["ttft_ms"]["median"],
                "decode_ms_median": r["decode_per_token_ms"]["median"],
                "throughput_median": r["throughput_tok_per_s"]["median"],
                "total_ms_median": r["total_ms"]["median"],
            })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = load_all()
    out = Path("results/merged.csv")
    df.to_csv(out, index=False)
    print(f"[saved] {out}  ({len(df)} rows)")
    print(df)
