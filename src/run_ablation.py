import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run import evaluate
from metrics import compute_all_metrics

RESULTS_DIR = Path("results")

K_VALUES = [1, 3, 5, 7]
MODEL = "openai/gpt-4o-mini"
MAX_SAMPLES = 200

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    print("=" * 60)
    print("ABLAÇÃO — NÚMERO DE EXEMPLOS FEW-SHOT (k)")
    print("=" * 60)
    print(f"Modelo: {MODEL}")
    print(f"Amostras: {MAX_SAMPLES}")
    print(f"k = {K_VALUES}")
    print()

    for k in K_VALUES:
        print(f"\n--- k={k} ---")
        evaluate(
            max_samples=MAX_SAMPLES,
            strategies=["few_shot"],
            model=MODEL,
            openrouter_key=openrouter_key,
            k=k,
        )

    model_slug = MODEL.replace("/", "-").replace(".", "-")
    print("\n" + "=" * 60)
    print("TABELA COMPARATIVA")
    print("=" * 60)

    rows = []
    for k in K_VALUES:
        path = RESULTS_DIR / model_slug / f"few_shot_k={k}_report.json"
        if not path.exists():
            print(f"  [AVISO] {path} não encontrado")
            continue
        with open(path) as f:
            report = json.load(f)
        m = report["metrics"]
        rows.append({
            "k": k,
            "R1": m.get("rouge1", "-"),
            "SARI": m.get("sari", "-"),
            "FΔ": m.get("flesch_improvement", "-"),
            "FogΔ": m.get("fog_improvement", "-"),
            "R1_OP": m.get("ori_pred_rouge1", "-"),
            "BS_OP": m.get("ori_pred_bertscore_mean", "-"),
        })

    header = f"{'k':<4} {'R1':<8} {'SARI':<8} {'FΔ':<10} {'FogΔ':<10} {'R1_OP':<8} {'BS_OP':<8}"
    print(header)
    print("-" * len(header))
    for r in rows:
        fd = f"{r['FΔ']:<+10.2f}" if isinstance(r['FΔ'], (int, float)) else f"{r['FΔ']:<10}"
        fg = f"{r['FogΔ']:<+10.2f}" if isinstance(r['FogΔ'], (int, float)) else f"{r['FogΔ']:<10}"
        print(f"{r['k']:<4} {r['R1']:<8} {r['SARI']:<8} {fd} {fg} {r['R1_OP']:<8} {r['BS_OP']:<8}")

    ablation_path = RESULTS_DIR / "k_ablation.json"
    with open(ablation_path, "w") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em: {ablation_path}")
