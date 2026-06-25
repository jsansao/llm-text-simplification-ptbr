import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_pairs

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"

DIMENSIONS = ["simplicidade", "fluencia", "adequacao"]
DIM_LABELS = {"simplicidade": "Simplicidade", "fluencia": "Fluência", "adequacao": "Adequação"}

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

MODEL_ORDER = list(MODEL_DISPLAY.keys())


def load_results() -> list[dict]:
    path = RESULTS_DIR / "llm_judge_results.json"
    if not path.exists():
        print(f"Arquivo não encontrado: {path}")
        sys.exit(1)
    with open(path) as f:
        return json.load(f)


def compute_mean_ratings(results: list[dict]) -> dict:
    by_model = defaultdict(lambda: {d: [] for d in DIMENSIONS})
    for r in results:
        if r["type"] != "model":
            continue
        model = r["model"]
        for d in DIMENSIONS:
            by_model[model][d].append(r["ratings"][d])

    means = {}
    for model, dim_vals in by_model.items():
        means[model] = {}
        for d in DIMENSIONS:
            vals = dim_vals[d]
            if vals:
                means[model][d] = round(statistics.mean(vals), 2)
                means[model][d + "_std"] = round(statistics.stdev(vals), 2)
            else:
                means[model][d] = 0.0
                means[model][d + "_std"] = 0.0
    return means


def compute_run_consistency(results: list[dict]) -> dict:
    # Per-model, per-dimension: std of run-level means
    by_model_run = defaultdict(lambda: defaultdict(lambda: {d: [] for d in DIMENSIONS}))
    for r in results:
        if r["type"] != "model":
            continue
        for d in DIMENSIONS:
            by_model_run[r["model"]][r["run"]][d].append(r["ratings"][d])

    consistency = {}
    for model in MODEL_ORDER:
        consistency[model] = {}
        for d in DIMENSIONS:
            run_means = []
            for run in range(3):
                vals = by_model_run[model][run].get(d, [])
                if vals:
                    run_means.append(statistics.mean(vals))
            if len(run_means) >= 2:
                consistency[model][d] = round(statistics.stdev(run_means), 3)
            else:
                consistency[model][d] = 0.0
    return consistency


def compute_sanity(results: list[dict]) -> dict:
    sanity = [r for r in results if r["type"] == "sanity"]
    dim_scores = {d: [] for d in DIMENSIONS}
    for r in sanity:
        for d in DIMENSIONS:
            dim_scores[d].append(r["ratings"][d])
    result = {}
    for d in DIMENSIONS:
        vals = dim_scores[d]
        if vals:
            result[d] = {"mean": round(statistics.mean(vals), 2), "std": round(statistics.stdev(vals), 2)}
        else:
            result[d] = {"mean": 0.0, "std": 0.0}
    return result


def load_auto_metrics() -> dict:
    df = load_pairs(max_samples=200, random_state=42)
    sources = [s.strip() for s in df["original"].tolist()]

    seen = set()
    unique_idxs = []
    for i, src in enumerate(sources):
        s = src.strip()
        if s not in seen:
            seen.add(s)
            unique_idxs.append(i)

    import evaluate
    rouge = evaluate.load("rouge")

    # Get predictions for each model
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
        path = RESULTS_DIR / f"ft-{model_name}" / "predictions.json"
        with open(path) as f:
            data = json.load(f)
        return {src.strip(): pred.strip() for src, pred in zip(data["sources"], data["predictions"])}

    best_strategy = {
        "openai/gpt-4o-mini": "few_shot",
        "openai/gpt-4o-2024-11-20": "few_shot",
        "anthropic/claude-sonnet-4.6": "few_shot",
        "meta-llama/llama-4-scout-17b-16e-instruct": "few_shot",
        "qwen/qwen-2.5-7b-instruct": "few_shot",
    }

    metrics = {}
    for model in MODEL_ORDER:
        if model in best_strategy:
            strat = best_strategy[model]
            entries = load_cache_preds(model, strat)
        else:
            name_map = {"ptt5-rank8": "ptt5", "ptt5-rank32": "ptt5-rank32", "ptt5-full": "ptt5-full", "qwen05b": "qwen"}
            entries = load_ft_preds(name_map[model])

        for i in unique_idxs:
            src_s = sources[i].strip()
            ref = df["simplified"].iloc[i]
            pred = entries.get(src_s, "")
            if pred:
                r = rouge.compute(predictions=[pred], references=[ref])
                metrics[(i, model, "rouge1")] = r["rouge1"]
            else:
                metrics[(i, model, "rouge1")] = 0.0
    return metrics


def compute_spearman(results: list[dict], auto_metrics: dict) -> dict:
    # Average LLM-judge ratings across runs per (sample, model)
    llm_by_pair = defaultdict(lambda: {d: [] for d in DIMENSIONS})
    for r in results:
        if r["type"] != "model":
            continue
        key = (r["idx"], r["model"])
        for d in DIMENSIONS:
            llm_by_pair[key][d].append(r["ratings"][d])

    llm_avg = {}
    for (idx, model), dim_vals in llm_by_pair.items():
        llm_avg[(idx, model)] = {d: statistics.mean(vals) for d, vals in dim_vals.items()}

    spearman_results = {}
    for d in DIMENSIONS:
        x_vals, y_vals = [], []
        for (idx, model), ratings in llm_avg.items():
            hv = ratings[d]
            av = auto_metrics.get((idx, model, "rouge1"))
            if hv is not None and av is not None:
                x_vals.append(hv)
                y_vals.append(av)
        if len(x_vals) >= 3:
            rho, p = spearmanr(x_vals, y_vals)
            spearman_results[d] = {"rho": round(rho, 3), "p": round(p, 4)}
        else:
            spearman_results[d] = {"rho": float("nan"), "p": float("nan")}
    return spearman_results


def compute_model_ranking(means: dict) -> dict:
    rankings = {}
    for d in DIMENSIONS:
        sorted_models = sorted(MODEL_ORDER, key=lambda m: means.get(m, {}).get(d, 0), reverse=True)
        rankings[d] = [(m, means[m][d]) for m in sorted_models]
    return rankings


def print_latex_tables(means: dict, consistency: dict, sanity: dict, spearman: dict, rankings: dict):
    print("\n% === TABELA 1: MÉDIAS LLM-JUDGE POR MODELO === %")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Avaliação LLM-as-judge (Gemini 3.1 Flash Lite) — médias por modelo (Likert 1--5, 3 execuções).}")
    cols = "l" + "c" * (3 * 2)
    print(r"\begin{tabular}{" + cols + r"}")
    print(r"\toprule")
    header = r"Modelo"
    for d in DIMENSIONS:
        header += r" & \multicolumn{2}{c}{" + DIM_LABELS[d] + r"}"
    print(header + r" \\")
    subheader = r""
    for _ in DIMENSIONS:
        subheader += r" & $M$ & $\sigma$"
    print(subheader + r" \\")
    print(r"\midrule")
    for model in MODEL_ORDER:
        if model not in means:
            continue
        line = MODEL_DISPLAY[model]
        for d in DIMENSIONS:
            m = means[model].get(d, 0)
            s = means[model].get(d + "_std", 0)
            line += f" & {m} & {s}"
        print(line + r" \\")
    print(r"\midrule")
    # Sanity check row
    line = r"Referência Humana (sanity)"
    for d in DIMENSIONS:
        m = sanity.get(d, {}).get("mean", 0)
        s = sanity.get(d, {}).get("std", 0)
        line += f" & {m} & {s}"
    print(line + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n% === TABELA 2: CONSISTÊNCIA ENTRE RUNS === %")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Consistência entre execuções — desvio padrão das médias por run.}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"Modelo & Simplicidade & Fluência & Adequação \\")
    print(r"\midrule")
    for model in MODEL_ORDER:
        if model not in consistency:
            continue
        line = MODEL_DISPLAY[model]
        for d in DIMENSIONS:
            line += f" & {consistency[model].get(d, 0):.3f}"
        print(line + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n% === TABELA 3: RANKING POR DIMENSÃO === %")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Ranking dos modelos por dimensão (LLM-as-judge).}")
    cols = "l" + "c" * len(DIMENSIONS)
    print(r"\begin{tabular}{" + cols + r"}")
    print(r"\toprule")
    header = r"Rank"
    for d in DIMENSIONS:
        header += r" & " + DIM_LABELS[d]
    print(header + r" \\")
    print(r"\midrule")
    for rank in range(len(MODEL_ORDER)):
        line = str(rank + 1)
        for d in DIMENSIONS:
            models_ranked = rankings.get(d, [])
            if rank < len(models_ranked):
                m, score = models_ranked[rank]
                line += f" & {MODEL_DISPLAY[m]} ({score})"
            else:
                line += " & ---"
        print(line + r" \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

    print("\n% === TABELA 4: CORRELAÇÃO SPEARMAN (LLM-JUDGE × ROUGE-1) === %")
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Correlação de Spearman entre LLM-judge e ROUGE-1.}")
    print(r"\begin{tabular}{lcc}")
    print(r"\toprule")
    print(r"Dimensão & $\rho$ & $p$ \\")
    print(r"\midrule")
    for d in DIMENSIONS:
        s = spearman.get(d, {"rho": float("nan"), "p": float("nan")})
        p_str = f"{s['p']:.4f}" if not np.isnan(s['p']) else "---"
        rho_str = f"{s['rho']:.3f}" if not np.isnan(s['rho']) else "---"
        print(f"{DIM_LABELS[d]} & {rho_str} & {p_str} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def main():
    results = load_results()
    print(f"Total de avaliações carregadas: {len(results)}")

    n_model = sum(1 for r in results if r["type"] == "model")
    n_sanity = sum(1 for r in results if r["type"] == "sanity")
    n_runs = len(set(r["run"] for r in results))
    print(f"  Modelos: {n_model} ({n_runs} runs)")
    print(f"  Sanity check: {n_sanity}")

    means = compute_mean_ratings(results)
    consistency = compute_run_consistency(results)
    sanity = compute_sanity(results)
    auto_metrics = load_auto_metrics()
    spearman = compute_spearman(results, auto_metrics)
    rankings = compute_model_ranking(means)

    print("\n=== MÉDIAS LLM-JUDGE POR MODELO ===")
    for model in MODEL_ORDER:
        if model not in means:
            continue
        line = f"  {MODEL_DISPLAY[model]}: "
        for d in DIMENSIONS:
            m = means[model][d]
            s = means[model][d + "_std"]
            line += f"{d}={m}±{s}  "
        print(line)

    print("\n=== SANITY CHECK (Referência Humana) ===")
    for d in DIMENSIONS:
        m = sanity.get(d, {}).get("mean", 0)
        s = sanity.get(d, {}).get("std", 0)
        status = "OK" if m >= 4.0 else "ATENÇÃO"
        print(f"  {d}: {m}±{s} [{status}]")

    print("\n=== CONSISTÊNCIA ENTRE RUNS ===")
    for model in MODEL_ORDER:
        if model not in consistency:
            continue
        line = f"  {MODEL_DISPLAY[model]}: "
        for d in DIMENSIONS:
            line += f"{d}={consistency[model][d]:.3f}  "
        print(line)

    print("\n=== RANKING POR DIMENSÃO ===")
    for d in DIMENSIONS:
        line = f"  {DIM_LABELS[d]}: "
        for rank, (m, score) in enumerate(rankings.get(d, []), 1):
            line += f"{rank}º {MODEL_DISPLAY[m]}({score})  "
        print(line)

    print("\n=== SPEARMAN (LLM-JUDGE × ROUGE-1) ===")
    for d in DIMENSIONS:
        s = spearman.get(d, {"rho": float("nan"), "p": float("nan")})
        print(f"  {d}: ρ={s['rho']}, p={s['p']}")

    print_latex_tables(means, consistency, sanity, spearman, rankings)

    # Save report
    report = {
        "means": means,
        "consistency": consistency,
        "sanity": sanity,
        "spearman_rouge1": spearman,
        "rankings": rankings,
    }
    report_path = RESULTS_DIR / "llm_judge_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nRelatório salvo em {report_path}")


if __name__ == "__main__":
    main()
