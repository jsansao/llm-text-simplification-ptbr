import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache import cache_key, cache_load, cache_save
from dataset import download_dataset, load_pairs
from llm_client import LLMClient
from metrics import compute_all_metrics
from prompts import (
    build_zero_shot,
    build_few_shot,
    build_cot,
    extract_simplification,
)

RESULTS_DIR = Path("results")
CACHE_DIR = Path("cache")
STRATEGIES = ["zero_shot", "few_shot", "cot"]


def _setup():
    download_dataset()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _get_examples(df, k: int = 3) -> list[tuple[str, str]]:
    sampled = df.sample(n=k, random_state=42)
    return list(zip(sampled["original"], sampled["simplified"]))


def run_strategy(
    strategy: str,
    df,
    client: LLMClient,
    max_samples: Optional[int] = None,
    examples: Optional[list[tuple[str, str]]] = None,
    cache_dir: Optional[Path] = None,
    no_cache: bool = False,
) -> list[dict]:
    results = []
    data = df
    n_new = 0
    n_cached = 0

    for idx, row in tqdm(data.iterrows(), total=len(data), desc=f"{strategy}"):
        original = row["original"]
        reference = row["simplified"]

        key = cache_key(original, client.model, strategy, client.temperature) if cache_dir else None

        if not no_cache and key and cache_dir:
            entry = cache_load(cache_dir, key)
            if entry:
                results.append({
                    "idx": int(idx),
                    "original": original,
                    "reference": reference,
                    "prediction": entry["prediction"],
                    "raw_output": entry["raw_output"],
                })
                n_cached += 1
                continue

        if strategy == "zero_shot":
            prompt = build_zero_shot(original)
        elif strategy.startswith("few_shot"):
            prompt = build_few_shot(original, examples)
        elif strategy == "cot":
            prompt = build_cot(original)
        else:
            raise ValueError(f"Estratégia desconhecida: {strategy}")

        raw_output = client.generate(prompt, strategy=strategy)
        prediction = extract_simplification(raw_output, strategy)

        if key and cache_dir:
            cache_save(cache_dir, key, {
                "raw_output": raw_output,
                "prediction": prediction,
                "original": original,
                "model": client.model,
                "strategy": strategy,
            })

        results.append({
            "idx": int(idx),
            "original": original,
            "reference": reference,
            "prediction": prediction,
            "raw_output": raw_output,
        })
        n_new += 1

    tqdm.write(f"  [{strategy}] {n_cached} em cache, {n_new} novos")
    return results


def evaluate(max_samples: int = 50, strategies: Optional[list[str]] = None,
             openrouter_key: Optional[str] = None, model: Optional[str] = None,
             cache_dir: Optional[Path] = None, no_cache: bool = False,
             k: int = 3):
    _setup()

    if strategies is None:
        strategies = STRATEGIES

    df = load_pairs(only_changed=True, max_samples=max_samples, random_state=42)
    print(f"[evaluate] {len(df)} pares ORI->NAT carregados (amostragem estratificada, seed=42)\n")

    client = LLMClient(temperature=0.0, max_tokens=512, openrouter_key=openrouter_key, model=model)
    examples = _get_examples(df, k=k)

    def _eff_strategy(s: str) -> str:
        return f"few_shot_k={k}" if s == "few_shot" else s

    if cache_dir and not no_cache:
        eff_strategies = [_eff_strategy(s) for s in strategies]
        hits = sum(1 for s in eff_strategies
                   for idx in range(min(max_samples or len(df), len(df)))
                   if cache_load(cache_dir, cache_key(df.iloc[idx]["original"], client.model, s, 0.0)))
        if hits:
            print(f"[cache] {hits} entradas encontradas, reutilizando\n")

    all_reports = {}

    for strategy in strategies:
        eff_strategy_name = _eff_strategy(strategy)

        print(f"\n{'='*60}")
        print(f"Estratégia: {eff_strategy_name}")
        print(f"{'='*60}")

        results = run_strategy(eff_strategy_name, df, client, examples=examples,
                               max_samples=max_samples, cache_dir=cache_dir, no_cache=no_cache)

        sources = [r["original"] for r in results]
        predictions = [r["prediction"] for r in results]
        references = [r["reference"] for r in results]

        metrics = compute_all_metrics(sources, predictions, references)
        print(f"\nMétricas ({eff_strategy_name}):")
        for metric_key, v in metrics.items():
            print(f"  {metric_key}: {v}")

        report = {
            "strategy": eff_strategy_name,
            "num_samples": len(results),
            "metrics": metrics,
            "results": results,
        }

        model_slug = client.model.replace("/", "-").replace(".", "-")
        model_dir = RESULTS_DIR / model_slug
        model_dir.mkdir(parents=True, exist_ok=True)
        path = model_dir / f"{eff_strategy_name}_report.json"
        with open(path, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        all_reports[strategy] = report

    print("\n" + "=" * 60)
    print("RESUMO COMPARATIVO")
    print("=" * 60)

    m0 = all_reports[strategies[0]]["metrics"]
    print(f"\nBaseline (Ori↔Ref): R1={m0.get('ori_ref_rouge1','-'):<8} R2={m0.get('ori_ref_rouge2','-'):<8} RL={m0.get('ori_ref_rougeL','-'):<8} BS={m0.get('ori_ref_bertscore_mean','-'):<8}")
    print(f"                     FleschΔ_ref={m0.get('ref_flesch_improvement','-'):>+7}  FogΔ_ref={m0.get('ref_fog_improvement','-'):>+7}\n")

    header = f"{'Estratégia':<12} {'R1':<7} {'R2':<7} {'RL':<7} {'BS':<7} {'SARI':<7} {'R1_OP':<7} {'BS_OP':<7} {'FΔ':<7} {'FogΔ':<7}"
    print(header)
    print("-" * len(header))
    for strategy in strategies:
        if strategy not in all_reports:
            continue
        m = all_reports[strategy]["metrics"]
        row = (f"{strategy:<12} "
               f"{m.get('rouge1', '-'):<7} {m.get('rouge2', '-'):<7} {m.get('rougeL', '-'):<7} "
               f"{m.get('bertscore_mean', '-'):<7} {m.get('sari', '-'):<7} "
               f"{m.get('ori_pred_rouge1', '-'):<7} {m.get('ori_pred_bertscore_mean', '-'):<7} "
               f"{m.get('flesch_improvement', '-'):<7} {m.get('fog_improvement', '-'):<7}")
        print(row)

    model_slug = client.model.replace("/", "-").replace(".", "-")
    model_dir = RESULTS_DIR / model_slug
    summary_path = model_dir / "summary.json"
    summary = {
        s: {"metrics": all_reports[s]["metrics"], "num_samples": all_reports[s]["num_samples"]}
        for s in strategies if s in all_reports
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    model_slug = client.model.replace("/", "-").replace(".", "-")
    print(f"\nResultados salvos em: {RESULTS_DIR.resolve() / model_slug}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avaliação de simplificação textual")
    parser.add_argument("--max-samples", type=int, default=50, help="Número de amostras (default: 50)")
    parser.add_argument("--strategies", nargs="+", default=STRATEGIES, choices=STRATEGIES,
                        help=f"Estratégias a executar (default: todas)")
    parser.add_argument("--model", type=str, default=None,
                        help="Modelo a usar (ex: openai/gpt-4o-mini, mimo-v2.5-free)")
    parser.add_argument("--openrouter-key", type=str, default=None,
                        help="Chave de API do OpenRouter (ou via OPENROUTER_API_KEY no .env)")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Diretório de cache (default: cache/)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Desabilitar cache")
    parser.add_argument("--k", type=int, default=3,
                        help="Número de exemplos few-shot (default: 3)")
    args = parser.parse_args()
    openrouter_key = args.openrouter_key or os.getenv("OPENROUTER_API_KEY")
    cache_dir = Path(args.cache_dir) if args.cache_dir else CACHE_DIR
    evaluate(max_samples=args.max_samples, strategies=args.strategies,
             openrouter_key=openrouter_key, model=args.model,
             cache_dir=cache_dir, no_cache=args.no_cache, k=args.k)
