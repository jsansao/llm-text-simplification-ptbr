"""Análise de sensibilidade: par #16 (vacinação vs. servidora pública)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate
from dataset import load_pairs

rouge = evaluate.load("rouge")
sari = evaluate.load("sari")


def metrics_n(idx_exclude, sources, preds, refs):
    if idx_exclude is None:
        s = sources
        p = preds
        r = refs
    else:
        s = [x for i, x in enumerate(sources) if i != idx_exclude]
        p = [x for i, x in enumerate(preds) if i != idx_exclude]
        r = [x for i, x in enumerate(refs) if i != idx_exclude]
    r1 = rouge.compute(predictions=p, references=r)["rouge1"]
    sa = sari.compute(sources=s, predictions=p, references=[[x] for x in r])["sari"]
    return r1, sa


def main():
    df = load_pairs(max_samples=200, random_state=42)
    sources = df["original"].tolist()
    refs = df["simplified"].tolist()
    print(f"Total: {len(sources)}")
    print(f"Amostra #16 (verificação):")
    print(f"  ORI: {sources[16][:100]}...")
    print(f"  REF: {refs[16][:100]}...")

    cache_dir = Path("cache")
    cache = {}
    for fn in cache_dir.glob("*.json"):
        with open(fn) as f:
            entry = json.load(f)
        if "model" in entry and "strategy" in entry and "original" in entry and "prediction" in entry:
            key = (entry["model"], entry["strategy"])
            cache.setdefault(key, {})[entry["original"].strip()] = entry["prediction"].strip()

    ft_dir = Path("results")
    for sub in ["ft-ptt5", "ft-ptt5-rank32", "ft-ptt5-full", "ft-qwen"]:
        path = ft_dir / sub / "predictions.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
            model_map = {
                "ft-ptt5": "ptt5",
                "ft-ptt5-rank32": "ptt5-rank32",
                "ft-ptt5-full": "ptt5-full",
                "ft-qwen": "qwen",
            }
            model_id = model_map[sub]
            key = (model_id, "FT")
            cache.setdefault(key, {})
            for s, p in zip(data["sources"], data["predictions"]):
                cache[key][s.strip()] = p.strip()

    models_to_test = [
        ("anthropic/claude-sonnet-4.6", "few_shot", "Claude S. 4.6 (few-shot)"),
        ("qwen/qwen-2.5-7b-instruct", "few_shot", "Qwen 2.5 7B (few-shot)"),
        ("qwen", "FT", "Qwen 0.5B FT"),
    ]

    print(f"\n{'Modelo':<28} {'R-1 (n=200)':>12} {'R-1 (n=199)':>12} {'Δ R-1':>8} {'SARI (200)':>11} {'SARI (199)':>11} {'Δ SARI':>8}")
    for model_id, strat, label in models_to_test:
        key = (model_id, strat)
        if key not in cache:
            print(f"{label:<28} — não encontrada")
            continue
        preds = []
        for s in sources:
            preds.append(cache[key].get(s.strip(), s.strip()))
        r1_200, sa_200 = metrics_n(None, sources, preds, refs)
        r1_199, sa_199 = metrics_n(16, sources, preds, refs)
        print(f"{label:<28} {r1_200:>12.4f} {r1_199:>12.4f} {r1_199-r1_200:>+8.4f} {sa_200:>11.2f} {sa_199:>11.2f} {sa_199-sa_200:>+8.2f}")


if __name__ == "__main__":
    main()
