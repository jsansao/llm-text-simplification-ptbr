import json
import os
import statistics

import evaluate

from dataset import load_pairs

CACHE_DIR = "cache"
RESULTS_DIR = "results"

BEST_STRATEGY = {
    "openai/gpt-4o-mini": "few_shot",
    "openai/gpt-4o-2024-11-20": "few_shot",
    "anthropic/claude-sonnet-4.6": "few_shot",
    "meta-llama/llama-4-scout-17b-16e-instruct": "few_shot",
    "qwen/qwen-2.5-7b-instruct": "few_shot",
}

MODEL_DISPLAY = {
    "openai/gpt-4o-mini": "GPT-4o mini",
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "anthropic/claude-sonnet-4.6": "Claude S. 4.6",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "qwen/qwen-2.5-7b-instruct": "Qwen 2.5 7B",
    "ptt5-rank8": "PTT5 LoRA r=8",
    "ptt5-rank32": "PTT5 LoRA r=32",
    "ptt5-full": "PTT5 full FT",
    "qwen05b": "Qwen 2.5 0.5B FT",
}

ALL_MODELS = list(BEST_STRATEGY.keys()) + ["ptt5-rank8", "ptt5-rank32", "ptt5-full", "qwen05b"]


def load_cache_preds(model: str, strategy: str) -> dict[str, str]:
    entries = {}
    for fn in os.listdir(CACHE_DIR):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(CACHE_DIR, fn)) as f:
            entry = json.load(f)
        if entry.get("model") == model and entry.get("strategy") == strategy:
            src = entry.get("original", "").strip()
            pred = entry.get("prediction", entry.get("raw_output", "")).strip()
            if src and pred:
                entries[src] = pred
    return entries


def load_ft_preds(model_name: str) -> dict[str, str]:
    path = os.path.join(RESULTS_DIR, f"ft-{model_name}", "predictions.json")
    with open(path) as f:
        data = json.load(f)
    return {src.strip(): pred.strip() for src, pred in zip(data["sources"], data["predictions"])}


def main():
    df = load_pairs(max_samples=200, random_state=42)
    sources = df["original"].tolist()
    refs = df["simplified"].tolist()

    all_preds = {}
    for model in ALL_MODELS:
        if model in BEST_STRATEGY:
            strat = BEST_STRATEGY[model]
            entries = load_cache_preds(model, strat)
        else:
            name_map = {"ptt5-rank8": "ptt5", "ptt5-rank32": "ptt5-rank32", "ptt5-full": "ptt5-full", "qwen05b": "qwen"}
            entries = load_ft_preds(name_map[model])
        preds = []
        found = 0
        for src in sources:
            src_s = src.strip()
            if src_s in entries:
                preds.append(entries[src_s])
                found += 1
            else:
                preds.append("")
        all_preds[model] = preds
        print(f"  [{MODEL_DISPLAY[model]}] {found}/{len(sources)} loaded")

    rouge = evaluate.load("rouge")

    per_sample_r1 = {}
    for model in ALL_MODELS:
        r1s = []
        for i, ref in enumerate(refs):
            p = all_preds[model][i]
            if not p:
                r1s.append(0.0)
            else:
                result = rouge.compute(predictions=[p], references=[ref])
                r1s.append(result["rouge1"])
        per_sample_r1[model] = r1s

    sample_stats = []
    for i in range(len(sources)):
        r1_vals = [per_sample_r1[m][i] for m in ALL_MODELS]
        mean_r1 = statistics.mean(r1_vals)
        std_r1 = statistics.stdev(r1_vals)
        sample_stats.append((i, mean_r1, std_r1, len(sources[i].split())))

    # 1: highest mean R1 (consensus)
    by_mean = sorted(sample_stats, key=lambda x: -x[1])
    s1 = by_mean[0]
    # 2: lowest mean R1 (hardest)
    s2 = by_mean[-1]
    # 3, 4: highest std R1 (divergence)
    by_std = sorted(sample_stats, key=lambda x: -x[2])
    s3 = by_std[0]
    s4 = by_std[1] if by_std[1][0] != s1[0] and by_std[1][0] != s2[0] else by_std[2]
    # 5: longest original text
    by_len = sorted(sample_stats, key=lambda x: -x[3])
    s5 = by_len[0]

    selected = [("Consenso (maior R1)", s1),
                ("Difícil (menor R1)", s2),
                ("Divergência #1 (maior std)", s3),
                ("Divergência #2 (2ª maior std)", s4),
                ("Mais longo", s5)]

    print("\n" + "=" * 80)
    print("AMOSTRAS SELECIONADAS")
    print("=" * 80)

    for label, (idx, mean_r1, std_r1, n_tokens) in selected:
        print(f"\n--- #{idx}: {label} (R1 médio={mean_r1:.3f}, std={std_r1:.3f}, tokens={n_tokens}) ---")
        print(f"  Original:   {sources[idx]}")
        print(f"  Referência: {refs[idx]}")
        for model in ALL_MODELS:
            p = all_preds[model][idx]
            r1 = per_sample_r1[model][idx]
            display = MODEL_DISPLAY[model]
            if p:
                print(f"  [{r1:.3f}] {display}: {p}")
            else:
                print(f"  [ N/A] {display}: (não disponível)")

    N_SAMPLES = 5
    print("\n\n=== DADOS PARA APÊNDICE LaTeX ===\n")
    for label, (idx, mean_r1, std_r1, n_tokens) in selected:
        print(f"% --- Amostra {selected.index((label, (idx, mean_r1, std_r1, n_tokens)))+1}: {label} ---")
        print(f"  \\textbf{{Original}}: & \\texttt{{{sources[idx][:400]}}} \\\\")
        print(f"  \\textbf{{Referência}}: & \\texttt{{{refs[idx][:400]}}} \\\\")
        for model in ALL_MODELS:
            p = all_preds[model][idx]
            r1 = per_sample_r1[model][idx]
            display = MODEL_DISPLAY[model]
            if p:
                print(f"  \\textbf{{{display}}}: & \\texttt{{{p[:400]}}} \\\\")
        print()


if __name__ == "__main__":
    main()
