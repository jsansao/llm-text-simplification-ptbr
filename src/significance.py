import json
import os
import random

import evaluate
import numpy as np

from dataset import load_pairs
from readability import flesch_reading_ease, gunning_fog

CACHE_DIR = "cache"
RESULTS_DIR = "results"

ROUGE_ENGINE = None
SARI_ENGINE = None
BERTSCORE_ENGINE = None


def _rouge():
    global ROUGE_ENGINE
    if ROUGE_ENGINE is None:
        ROUGE_ENGINE = evaluate.load("rouge")
    return ROUGE_ENGINE


def _sari():
    global SARI_ENGINE
    if SARI_ENGINE is None:
        SARI_ENGINE = evaluate.load("sari")
    return SARI_ENGINE


def _bertscore():
    global BERTSCORE_ENGINE
    if BERTSCORE_ENGINE is None:
        BERTSCORE_ENGINE = evaluate.load("bertscore")
    return BERTSCORE_ENGINE


def load_dataset():
    df = load_pairs(max_samples=200, random_state=42)
    return list(zip(df["original"].tolist(), df["simplified"].tolist()))


def load_cache_entries(model: str, strategy: str) -> dict[str, str]:
    entries = {}
    for fn in os.listdir(CACHE_DIR):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(CACHE_DIR, fn)) as f:
            entry = json.load(f)
        if entry.get("model") == model and entry.get("strategy") == strategy:
            original = entry.get("original", "").strip()
            prediction = entry.get("prediction", entry.get("raw_output", "")).strip()
            if original and prediction:
                entries[original] = prediction
    return entries


def load_ft_predictions(model_name: str) -> dict[str, str]:
    path = os.path.join(RESULTS_DIR, f"ft-{model_name}", "predictions.json")
    with open(path) as f:
        data = json.load(f)
    entries = {}
    for src, pred in zip(data["sources"], data["predictions"]):
        entries[src.strip()] = pred.strip()
    return entries


PICKED_MODELS = [
    ("GPT-4o mini", "few_shot", lambda: load_cache_entries("openai/gpt-4o-mini", "few_shot")),
    ("GPT-4o", "few_shot", lambda: load_cache_entries("openai/gpt-4o-2024-11-20", "few_shot")),
    ("Claude S. 4.6", "few_shot", lambda: load_cache_entries("anthropic/claude-sonnet-4.6", "few_shot")),
    ("Qwen 2.5 7B", "few_shot", lambda: load_cache_entries("qwen/qwen-2.5-7b-instruct", "few_shot")),
    ("Llama 4 Scout", "few_shot", lambda: load_cache_entries("meta-llama/llama-4-scout-17b-16e-instruct", "few_shot")),
    ("Llama 4 Scout CoT", "CoT", lambda: load_cache_entries("meta-llama/llama-4-scout-17b-16e-instruct", "cot")),
    ("GPT-4o mini CoT", "CoT", lambda: load_cache_entries("openai/gpt-4o-mini", "cot")),
    ("GPT-4o CoT", "CoT", lambda: load_cache_entries("openai/gpt-4o-2024-11-20", "cot")),
    ("Qwen 2.5 7B CoT", "CoT", lambda: load_cache_entries("qwen/qwen-2.5-7b-instruct", "cot")),
    ("PTT5 LoRA r=8", "FT", lambda: load_ft_predictions("ptt5")),
    ("PTT5 LoRA r=32", "FT", lambda: load_ft_predictions("ptt5-rank32")),
    ("PTT5 full FT", "FT", lambda: load_ft_predictions("ptt5-full")),
    ("Qwen 2.5 0.5B FT", "FT", lambda: load_ft_predictions("qwen")),
]

PAIRS = [
    ("GPT-4o mini", "Claude S. 4.6"),
    ("GPT-4o mini", "GPT-4o"),
    ("GPT-4o mini", "Qwen 2.5 7B"),
    ("GPT-4o mini", "Llama 4 Scout"),
    ("GPT-4o mini", "PTT5 LoRA r=8"),
    ("GPT-4o mini", "PTT5 LoRA r=32"),
    ("GPT-4o mini", "PTT5 full FT"),
    ("GPT-4o mini", "Qwen 2.5 0.5B FT"),
    ("PTT5 LoRA r=8", "PTT5 LoRA r=32"),
    ("PTT5 LoRA r=8", "PTT5 full FT"),
    ("PTT5 LoRA r=32", "PTT5 full FT"),
    ("PTT5 LoRA r=8", "Qwen 2.5 0.5B FT"),
    ("PTT5 LoRA r=32", "Qwen 2.5 0.5B FT"),
    ("PTT5 full FT", "Qwen 2.5 0.5B FT"),
    ("Claude S. 4.6", "GPT-4o"),
    ("Qwen 2.5 7B", "Llama 4 Scout"),
    ("Llama 4 Scout", "Llama 4 Scout CoT"),
    ("GPT-4o mini", "GPT-4o mini CoT"),
    ("Llama 4 Scout CoT", "GPT-4o mini CoT"),
    ("GPT-4o", "GPT-4o CoT"),
    ("GPT-4o CoT", "Llama 4 Scout CoT"),
    ("GPT-4o CoT", "GPT-4o mini CoT"),
    ("Qwen 2.5 7B", "Qwen 2.5 7B CoT"),
    ("Qwen 2.5 7B CoT", "Llama 4 Scout CoT"),
    ("Qwen 2.5 7B CoT", "GPT-4o mini CoT"),
    ("Qwen 2.5 7B CoT", "GPT-4o CoT"),
]

N_BOOTSTRAP = 10000
SEED = 42


def per_sample_rouge1(preds: list[str], refs: list[str]) -> list[float]:
    scores = []
    for p, r in zip(preds, refs):
        result = _rouge().compute(predictions=[p], references=[r])
        scores.append(result["rouge1"])
    return scores


def per_sample_rouge1_op(sources: list[str], preds: list[str]) -> list[float]:
    scores = []
    for s, p in zip(sources, preds):
        result = _rouge().compute(predictions=[p], references=[s])
        scores.append(result["rouge1"])
    return scores


def per_sample_sari(sources: list[str], preds: list[str], refs: list[str]) -> list[float]:
    scores = []
    for s, p, r in zip(sources, preds, refs):
        result = _sari().compute(sources=[s], predictions=[p], references=[[r]])
        scores.append(result["sari"])
    return scores


def per_sample_flesch_delta(preds: list[str], sources: list[str]) -> list[float]:
    return [flesch_reading_ease(p) - flesch_reading_ease(s) for s, p in zip(sources, preds)]


def per_sample_fog_delta(preds: list[str], sources: list[str]) -> list[float]:
    return [gunning_fog(p) - gunning_fog(s) for s, p in zip(sources, preds)]


def per_sample_bertscore_op(sources: list[str], preds: list[str]) -> list[float]:
    result = _bertscore().compute(predictions=preds, references=sources, lang="pt")
    return result["f1"]


def bootstrap_test(a: list[float], b: list[float], n_iterations: int = N_BOOTSTRAP, seed: int = SEED) -> tuple[float, float, float]:
    rng = random.Random(seed)
    n = len(a)
    diffs = []
    for _ in range(n_iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(np.mean([a[i] for i in idx]) - np.mean([b[i] for i in idx]))
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    p_value = 2 * min(
        sum(1 for d in diffs if d >= 0) / n_iterations,
        sum(1 for d in diffs if d <= 0) / n_iterations,
    )
    return float(ci_low), float(ci_high), float(p_value)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    decisions = [False] * n
    for k, (idx, pv) in enumerate(indexed):
        threshold = alpha / (n - k)
        if pv < threshold:
            decisions[idx] = True
        else:
            break
    return decisions


def main():
    dataset = load_dataset()
    sources = [o for o, _ in dataset]
    references = [r for _, r in dataset]

    model_preds: dict[str, list[str]] = {}
    model_valid: dict[str, list[int]] = {}

    for name, strat, loader in PICKED_MODELS:
        entries = loader()
        preds = []
        valid = []
        for i, (src, ref) in enumerate(dataset):
            src = src.strip()
            if src in entries:
                preds.append(entries[src])
                valid.append(i)
            else:
                print(f"  [!] {name}: sample {i} not found")
        model_preds[name] = preds
        model_valid[name] = valid
        print(f"  [{name}] ({strat}) {len(preds)}/{len(dataset)} samples loaded")

    aligned = sorted(set.intersection(*[set(v) for v in model_valid.values()]))
    print(f"\nAligned samples: {len(aligned)}/{len(dataset)}")
    if len(aligned) < len(dataset):
        print("  Warning: not all samples available across all models")

    idx = aligned
    models = [name for name, _, _ in PICKED_MODELS]

    metrics_data = {}
    for name in models:
        p = [model_preds[name][model_valid[name].index(i)] for i in idx]
        ss = [sources[i] for i in idx]
        rs = [references[i] for i in idx]
        metrics_data[name] = {
            "R1": per_sample_rouge1(p, rs),
            "SARI": per_sample_sari(ss, p, rs),
            "F\u0394": per_sample_flesch_delta(p, ss),
            "Fog\u0394": per_sample_fog_delta(p, ss),
            "R1_OP": per_sample_rouge1_op(ss, p),
            "BS_OP": per_sample_bertscore_op(ss, p),
        }

    metric_labels = [
        ("R1", "R1", "{:.3f}"),
        ("SARI", "SARI", "{:.1f}"),
        ("F\u0394", "F\u0394", "{:+.2f}"),
        ("Fog\u0394", "Fog\u0394", "{:+.2f}"),
        ("R1_OP", "R1$_{OP}$", "{:.3f}"),
        ("BS_OP", "BS$_{OP}$", "{:.3f}"),
    ]

    print(f"\n{'Modelo':<20}", end="")
    for _, label, _ in metric_labels:
        print(f"{label:>8}", end="")
    print()
    print("-" * 70)
    for name in models:
        print(f"{name:<20}", end="")
        for key, _, fmt in metric_labels:
            v = np.mean(metrics_data[name][key])
            print(f"{fmt.format(v):>8}", end="")
        print()

    print(f"\n--- Paired Bootstrap Tests (n={N_BOOTSTRAP}, \u03b1=0.05) ---")
    header = f"{'Par':<40} {'M\u00e9trica':>8} {'Diferen\u00e7a':>10} {'IC95%':>22} {'p':>8} {'Signif.':>8}"
    print(f"\n{header}")
    print("-" * 100)

    results = []
    for model_a, model_b in PAIRS:
        for key, label, _ in metric_labels:
            a = metrics_data[model_a][key]
            b = metrics_data[model_b][key]
            diff_mean = float(np.mean(a) - np.mean(b))
            ci_low, ci_high, p_value = bootstrap_test(a, b)
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
            print(f"{model_a+' vs '+model_b:<40} {label:>8} {diff_mean:>+10.4f} [{ci_low:>8.4f}, {ci_high:>8.4f}] {p_value:>8.4f} {sig:>8}")
            results.append({
                "pair": f"{model_a} vs {model_b}",
                "metric": label,
                "diff_mean": round(diff_mean, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
            })

    print("\n--- Tabela Resumo com Letras (7 colunas) ---")
    col_spec = "l " + " ".join(["c"] * len(metric_labels))
    header_cols = " & ".join(["Modelo"] + [lb for _, lb, _ in metric_labels])

    all_labels = {}
    holm_results = []
    for key, _, _ in metric_labels:
        pk = [(model_a, model_b) for model_a, model_b in PAIRS]
        a_list = [metrics_data[a][key] for a, _ in PAIRS]
        b_list = [metrics_data[b][key] for _, b in PAIRS]
        raw_p = [bootstrap_test(a, b)[2] for a, b in zip(a_list, b_list)]
        holm_decisions = holm_bonferroni(raw_p)

        group_map = {name: set() for name in models}
        for (ma, mb), sig_flag in zip(PAIRS, holm_decisions):
            if not sig_flag:
                group_map[ma].add(mb)
                group_map[mb].add(ma)

        groups = []
        assigned = set()
        for name in models:
            if name not in assigned:
                g = {name} | {m for m in group_map.get(name, set())}
                groups.append(g)
                assigned |= g
        lbl_map = {}
        for i, g in enumerate(groups):
            ch = chr(97 + i)
            for m in g:
                lbl_map[m] = lbl_map.get(m, "") + ch
        all_labels[key] = lbl_map

        for (ma, mb), pv, sig in zip(PAIRS, raw_p, holm_decisions):
            holm_results.append({
                "pair": f"{ma} vs {mb}",
                "metric": key,
                "p_value_raw": round(pv, 4),
                "holm_adjusted_alpha": True,
                "significant_holm": sig,
            })

    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{M\\'edias por modelo. Sobrescritos iguais = sem diferen\\c{c}a significativa (corre\\c{c}\\~ao de Holm-Bonferroni, $\\alpha = 0{,}05$, bootstrap pareado, 10.000 itera\\c{c}\\~oes).}")
    print("\\label{tab:significancia}")
    print("\\resizebox{\\textwidth}{!}{%")
    print("\\begin{tabular}{" + col_spec + "}")
    print("\\toprule")
    print(header_cols + " \\\\")
    print("\\midrule")

    for name in models:
        cells = []
        for key, _, fmt in metric_labels:
            v = np.mean(metrics_data[name][key])
            lbl = "".join(sorted(set(all_labels[key].get(name, "?"))))
            cells.append(f"${fmt.format(v)}^{{\\tiny {lbl}}}$")
        print(f"{name:<20} & " + " & ".join(cells) + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}%")
    print("}")
    print("\\end{table}")

    out_json = {"pairwise_tests": results, "holm_correction": holm_results}
    with open("results/significance.json", "w") as f:
        json.dump(out_json, f, indent=2, ensure_ascii=False)

    n_raw_sig = sum(1 for r in results if r["significant"])
    n_holm_sig = sum(1 for h in holm_results if h["significant_holm"])
    print(f"\nCorreção Holm-Bonferroni: {n_raw_sig} testes significantes (raw) → {n_holm_sig} (Holm, α=0.05)")
    print(f"Resultados salvos em: results/significance.json")


if __name__ == "__main__":
    main()
