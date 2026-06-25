"""Compute inter-judge agreement between Gemini and GPT-4o mini judges."""
import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

DIMENSIONS = ["simplicidade", "fluencia", "adequacao"]
MODEL_ORDER = [
    "openai/gpt-4o-mini",
    "openai/gpt-4o-2024-11-20",
    "anthropic/claude-sonnet-4.6",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen-2.5-7b-instruct",
    "ptt5-rank8",
    "ptt5-rank32",
    "ptt5-full",
    "qwen05b",
    "reference",
]
MODEL_DISPLAY = {
    "openai/gpt-4o-mini": "GPT-4o mini",
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "anthropic/claude-sonnet-4.6": "Claude S. 4.6",
    "meta-llama/llama-4-scout-17b-16e-instruct": "Llama 4 Scout",
    "qwen/qwen-2.5-7b-instruct": "Qwen 2.5 7B",
    "ptt5-rank8": "PTT5 r=8",
    "ptt5-rank32": "PTT5 r=32",
    "ptt5-full": "PTT5 full",
    "qwen05b": "Qwen 0.5B",
    "reference": "Humano",
}


def _pairs_from_flat(data):
    n = len(data)
    if n < 2 or n % 2 != 0:
        return []
    return [(data[2 * i], data[2 * i + 1]) for i in range(n // 2)]


def krippendorff_alpha(data):
    """Krippendorff's alpha for two judges, ordinal data (squared difference).

    Args:
        data: flat list [g1, l1, g2, l2, ..., gn, ln]

    Returns:
        alpha value
    """
    n = len(data)
    if n < 4 or n % 2 != 0:
        return float('nan')

    n_items = n // 2

    D_o = 0.0
    for i in range(n_items):
        g = data[2 * i]
        l = data[2 * i + 1]
        D_o += (g - l) ** 2
    D_o /= n_items

    N = n
    sum_v = sum(data)
    sum_v2 = sum(v * v for v in data)
    total_pair_diff = 2 * (N * sum_v2 - sum_v * sum_v)
    D_e = total_pair_diff / (N * (N - 1))

    if D_e == 0:
        return 1.0

    alpha = 1 - D_o / D_e
    return alpha


def krippendorff_alpha_general(ratings_matrix):
    """Krippendorff's alpha for M coders, N units, interval data.

    Args:
        ratings_matrix: list of lists, shape (N_units, M_coders)

    Returns:
        alpha value
    """
    if not ratings_matrix or len(ratings_matrix) < 2:
        return float('nan')

    N = len(ratings_matrix)
    M = len(ratings_matrix[0])
    if M < 2:
        return float('nan')

    all_values = []
    for row in ratings_matrix:
        all_values.extend(row)

    # Observed disagreement: average squared difference within each unit
    D_o = 0.0
    n_pairs = 0
    for row in ratings_matrix:
        for i in range(M):
            for j in range(i + 1, M):
                D_o += (row[i] - row[j]) ** 2
                n_pairs += 1
    D_o /= n_pairs if n_pairs > 0 else 1

    # Expected disagreement: average squared difference across all value pairs
    K = N * M
    sum_v = sum(all_values)
    sum_v2 = sum(v * v for v in all_values)
    total_pair_diff = 2 * (K * sum_v2 - sum_v * sum_v)
    D_e = total_pair_diff / (K * (K - 1)) if K > 1 else float('inf')

    if D_e == 0:
        return 1.0

    alpha = 1 - D_o / D_e
    return alpha


def cohens_kappa(data):
    """Cohen's Kappa (unweighted) for two judges.

    Args:
        data: flat list [g1, l1, g2, l2, ..., gn, ln]

    Returns:
        kappa value
    """
    pairs = _pairs_from_flat(data)
    if not pairs:
        return float('nan')
    n = len(pairs)

    max_cat = max(max(g, l) for g, l in pairs)
    k = max_cat + 1

    obs = [[0] * k for _ in range(k)]
    for g, l in pairs:
        obs[g][l] += 1

    row_sums = [sum(r) for r in obs]
    col_sums = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    p_o = sum(obs[i][i] for i in range(k)) / n
    p_e = sum(row_sums[i] * col_sums[i] for i in range(k)) / (n * n)

    if p_e >= 1:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def cohens_weighted_kappa(data):
    """Cohen's Weighted Kappa (linear weights) for two judges.

    Args:
        data: flat list [g1, l1, g2, l2, ..., gn, ln]

    Returns:
        weighted kappa value
    """
    pairs = _pairs_from_flat(data)
    if not pairs:
        return float('nan')
    n = len(pairs)

    max_cat = max(max(g, l) for g, l in pairs)
    k = max_cat + 1

    obs = [[0] * k for _ in range(k)]
    for g, l in pairs:
        obs[g][l] += 1

    row_sums = [sum(r) for r in obs]
    col_sums = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    w = [[1.0 - abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]

    expected = [[row_sums[i] * col_sums[j] / n for j in range(k)] for i in range(k)]

    weighted_obs = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    weighted_exp = sum(w[i][j] * expected[i][j] for i in range(k) for j in range(k))

    if weighted_exp == weighted_obs:
        return 1.0
    return 1 - (weighted_obs / weighted_exp)


def compute_agreement(gemini_data, llama_data):
    """Compute all agreement metrics between two judges for each (model, dimension) pair."""
    gemini_index = {}
    for r in gemini_data:
        gemini_index[(r["idx"], r["run"], r["model"])] = r["ratings"]

    llama_index = {}
    for r in llama_data:
        llama_index[(r["idx"], r["run"], r["model"])] = r["ratings"]

    common_keys = set(gemini_index.keys()) & set(llama_index.keys())

    by_model = defaultdict(lambda: {d: [] for d in DIMENSIONS})
    for key in common_keys:
        model = key[2]
        gemini_ratings = gemini_index[key]
        llama_ratings = llama_index[key]
        for d in DIMENSIONS:
            if d in gemini_ratings and d in llama_ratings:
                by_model[model][d].append((gemini_ratings[d], llama_ratings[d]))

    agreement = {}
    for model in MODEL_ORDER:
        if model not in by_model:
            continue
        agreement[model] = {}
        for d in DIMENSIONS:
            pairs = by_model[model][d]
            if not pairs:
                agreement[model][d] = {"alpha": float('nan'), "kappa": float('nan'), "kappa_w": float('nan')}
                continue
            values = []
            for g, l in pairs:
                values.append(g)
                values.append(l)
            agreement[model][d] = {
                "alpha": krippendorff_alpha(values),
                "kappa": cohens_kappa(values),
                "kappa_w": cohens_weighted_kappa(values),
            }

    return agreement


def compute_intra_agreement(data):
    """Compute intra-judge Krippendorff's alpha across 3 runs for each (model, dimension).

    Args:
        data: list of rating dicts from a single judge, each with idx, run, model, ratings

    Returns:
        dict: model -> dimension -> alpha
    """
    by_model_dim = defaultdict(lambda: defaultdict(list))
    for r in data:
        if r["type"] != "model":
            continue
        model = r["model"]
        idx = r["idx"]
        for d in DIMENSIONS:
            by_model_dim[(model, d)][idx].append(r["ratings"][d])

    intra = {}
    for model in MODEL_ORDER:
        if model == "reference":
            continue
        intra[model] = {}
        for d in DIMENSIONS:
            groups = list(by_model_dim.get((model, d), {}).values())
            # Only keep groups with all 3 runs
            groups = [g for g in groups if len(g) == N_RUNS]
            if len(groups) < 2:
                intra[model][d] = float('nan')
                continue
            intra[model][d] = krippendorff_alpha_general(groups)

    return intra


N_RUNS = 3


def compute_per_dimension_agreement(agreement, metric="alpha"):
    """Compute per-dimension and overall average for a specific metric."""
    dim_avgs = {}
    for d in DIMENSIONS:
        vals = [agreement[m][d][metric] for m in MODEL_ORDER if m in agreement and d in agreement[m]]
        vals = [v for v in vals if not (isinstance(v, float) and v != v)]
        dim_avgs[d] = sum(vals) / len(vals) if vals else float('nan')

    all_vals = []
    for m in MODEL_ORDER:
        if m in agreement:
            for d in DIMENSIONS:
                v = agreement[m][d].get(metric)
                if v is not None and not (isinstance(v, float) and v != v):
                    all_vals.append(v)
    overall_avg = sum(all_vals) / len(all_vals) if all_vals else float('nan')

    return dim_avgs, overall_avg


def avg_non_nan(vals):
    valid = [v for v in vals if not (isinstance(v, float) and v != v)]
    return sum(valid) / len(valid) if valid else float('nan')


def main():
    gemini_path = RESULTS_DIR / "llm_judge_results.json"
    llama_path = RESULTS_DIR / "llm_judge_results_gpt4omini.json"

    if not gemini_path.exists():
        print(f"ERROR: Gemini results not found at {gemini_path}")
        sys.exit(1)
    if not llama_path.exists():
        print(f"ERROR: Llama results not found at {llama_path}")
        sys.exit(1)

    with open(gemini_path) as f:
        gemini_data = json.load(f)
    with open(llama_path) as f:
        llama_data = json.load(f)

    print(f"Gemini judge: {len(gemini_data)} ratings")
    print(f"Llama judge:  {len(llama_data)} ratings")

    # Inter-judge agreement
    agreement = compute_agreement(gemini_data, llama_data)

    # Intra-judge agreement
    print("\nComputing intra-judge consistency (Krippendorff's α across 3 runs)...")
    intra_gemini = compute_intra_agreement(gemini_data)
    intra_llama = compute_intra_agreement(llama_data)

    print(f"\n=== Inter-judge Agreement: Gemini vs GPT-4o mini ===")

    # --- Print tables ---
    for metric_name, metric_key, desc in [
        ("Krippendorff's α", "alpha", "ordinal (squared difference)"),
        ("Cohen's κ (unweighted)", "kappa", "exact agreement"),
        ("Cohen's κw (linear)", "kappa_w", "ordinal, linear weights"),
    ]:
        dim_avgs, overall_avg = compute_per_dimension_agreement(agreement, metric_key)

        print(f"\n--- {metric_name} ({desc}) ---")
        print(f"{'Model':<20} {'Simplicidade':>12} {'Fluência':>10} {'Adequação':>10} {'Média':>8}")
        print("-" * 62)

        for model in MODEL_ORDER:
            if model not in agreement:
                continue
            scores = agreement[model]
            vals = [scores[d][metric_key] for d in DIMENSIONS]
            avg = avg_non_nan(vals)
            line = f"{MODEL_DISPLAY.get(model, model):<20}"
            for v in vals:
                if v != v:
                    line += f" {'N/A':>10}"
                else:
                    line += f" {v:>10.3f}"
            if avg == avg:
                line += f" {avg:>8.3f}"
            else:
                line += f" {'N/A':>8}"
            print(line)

        print("-" * 62)
        line = f"{'Média':<20}"
        for d in DIMENSIONS:
            v = dim_avgs.get(d, float('nan'))
            if v == v:
                line += f" {v:>10.3f}"
            else:
                line += f" {'N/A':>10}"
        if overall_avg == overall_avg:
            line += f" {overall_avg:>8.3f}"
        else:
            line += f" {'N/A':>8}"
        print(line)

    # Combined inter + intra table
    print(f"\n=== Combined Table (α / κ / κw / αi-G / αi-M) ===")
    header = f"{'Model':<20} {'α-Simp':>15} {'α-Flu':>15} {'α-Adeq':>15} {'κ':>10} {'κw':>10} {'αi-G':>10} {'αi-M':>10}"
    print(header)
    print("-" * 106)

    for model in MODEL_ORDER:
        if model not in agreement:
            continue
        line = f"{MODEL_DISPLAY.get(model, model):<20}"
        for d in DIMENSIONS:
            v = agreement[model][d]['alpha']
            if v == v:
                line += f" {v:>15.3f}"
            else:
                line += f" {'N/A':>15}"

        k = avg_non_nan([agreement[model][d]['kappa'] for d in DIMENSIONS])
        kw = avg_non_nan([agreement[model][d]['kappa_w'] for d in DIMENSIONS])
        if k == k:
            line += f" {k:>10.3f}"
        else:
            line += f" {'N/A':>10}"
        if kw == kw:
            line += f" {kw:>10.3f}"
        else:
            line += f" {'N/A':>10}"

        ig = avg_non_nan([intra_gemini.get(model, {}).get(d, float('nan')) for d in DIMENSIONS])
        im = avg_non_nan([intra_llama.get(model, {}).get(d, float('nan')) for d in DIMENSIONS])
        if ig == ig:
            line += f" {ig:>10.3f}"
        else:
            line += f" {'N/A':>10}"
        if im == im:
            line += f" {im:>10.3f}"
        else:
            line += f" {'N/A':>10}"
        print(line)

    # Intra-judge summary
    print(f"\n=== Intra-judge Consistency (α across 3 runs) ===")
    print(f"{'Model':<20} {'Gemini':>10} {'GPT-4o mini':>12}")
    print("-" * 44)
    for model in MODEL_ORDER:
        if model == "reference":
            continue
        if model not in intra_gemini and model not in intra_llama:
            continue
        ig = avg_non_nan([intra_gemini.get(model, {}).get(d, float('nan')) for d in DIMENSIONS])
        im = avg_non_nan([intra_llama.get(model, {}).get(d, float('nan')) for d in DIMENSIONS])
        line = f"{MODEL_DISPLAY.get(model, model):<20}"
        line += f" {ig:>10.3f}" if ig == ig else f" {'N/A':>10}"
        line += f" {im:>12.3f}" if im == im else f" {'N/A':>12}"
        print(line)

    gem_avg = avg_non_nan([avg_non_nan([intra_gemini[m][d] for d in DIMENSIONS]) for m in MODEL_ORDER if m != "reference" and m in intra_gemini])
    llam_avg = avg_non_nan([avg_non_nan([intra_llama[m][d] for d in DIMENSIONS]) for m in MODEL_ORDER if m != "reference" and m in intra_llama])
    print("-" * 44)
    print(f"{'Média':<20} {gem_avg:>10.3f} {llam_avg:>12.3f}")

    print(f"\nInterpretation: α/κ < 0 → no agreement, 0 ≤ < 0.2 → slight, "
          f"0.2 ≤ < 0.4 → fair, 0.4 ≤ < 0.6 → moderate, "
          f"0.6 ≤ < 0.8 → substantial, ≥ 0.8 → almost perfect")


if __name__ == "__main__":
    main()
