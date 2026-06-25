import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_pairs
from llm_client import LLMClient

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

BEST_STRATEGY = {
    "openai/gpt-4o-mini": "few_shot",
    "openai/gpt-4o-2024-11-20": "few_shot",
    "anthropic/claude-sonnet-4.6": "few_shot",
    "meta-llama/llama-4-scout-17b-16e-instruct": "few_shot",
    "qwen/qwen-2.5-7b-instruct": "few_shot",
}

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
]

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

JUDGE_MODEL = "google/gemini-3.1-flash-lite"
JUDGE_TEMPERATURE = 0.3
N_RUNS = 3

DIMENSIONS = ["simplicidade", "fluencia", "adequacao"]

EVAL_PROMPT = """Você é um avaliador de simplificação textual em português brasileiro.

Você receberá:
- ORIGINAL: o texto original (complexo)
- REF: uma simplificação de referência feita por um humano
- SISTEMA: uma simplificação automática a ser avaliada

Avalie o SISTEMA comparado ao ORIGINAL, usando a REF como referência de qualidade.
Responda APENAS com um objeto JSON válido com notas de 1 a 5 (Likert):

1 = Péssimo, 2 = Ruim, 3 = Regular, 4 = Bom, 5 = Excelente

Dimensões:
- "simplicidade": o texto SISTEMA é mais simples de ler que o ORIGINAL?
- "fluencia": o texto SISTEMA é natural e bem formado em português?
- "adequacao": o texto SISTEMA preserva o significado do ORIGINAL?

Exemplo de resposta:
{{"simplicidade": 4, "fluencia": 3, "adequacao": 5}}

ORIGINAL: {original}
REF: {reference}
SISTEMA: {prediction}"""


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


def judge_cache_key(original: str, model: str, run: int, prediction_hint: str = "") -> str:
    raw = f"llm_judge||{JUDGE_MODEL}||{JUDGE_TEMPERATURE}||run{run}||{model}||{original}"
    if prediction_hint:
        raw += f"||pred:{prediction_hint}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def parse_ratings(response: str) -> dict | None:
    response = response.strip()
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    response = response.strip()
    try:
        data = json.loads(response)
    except json.JSONDecodeError:
        import re
        match = re.search(r'\{[^}]+\}', response)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return None
        else:
            return None
    ratings = {}
    for dim in DIMENSIONS:
        val = data.get(dim)
        if isinstance(val, (int, float)) and 1 <= val <= 5:
            ratings[dim] = int(val)
    if len(ratings) == len(DIMENSIONS):
        return ratings
    return None


def evaluate(client: LLMClient, prompt: str, key: str, run: int) -> dict | None:
    cached = CACHE_DIR / f"{key}.json"
    if cached.exists():
        with open(cached) as f:
            return json.load(f).get("ratings")

    response = client.generate(prompt, strategy="cot")
    if not response:
        print(f"  [run {run}] Empty response")
        return None

    ratings = parse_ratings(response)
    if ratings is None:
        print(f"  [run {run}] Failed to parse: {response[:100]}")
        return None

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cached, "w") as f:
        json.dump({"prompt": prompt, "response": response, "ratings": ratings, "timestamp": time.time()}, f, ensure_ascii=False)
    return ratings


def main():
    print("Carregando dados...")
    df = load_pairs(max_samples=200, random_state=42)
    sources = df["original"].tolist()
    refs = df["simplified"].tolist()

    # Deduplicate by original text
    seen = set()
    unique_idxs = []
    for i, src in enumerate(sources):
        s = src.strip()
        if s not in seen:
            seen.add(s)
            unique_idxs.append(i)
    print(f"  {len(unique_idxs)} amostras únicas")

    # Load all predictions
    all_preds = {}
    for model in MODEL_ORDER:
        if model in BEST_STRATEGY:
            strat = BEST_STRATEGY[model]
            entries = load_cache_preds(model, strat)
        else:
            name_map = {"ptt5-rank8": "ptt5", "ptt5-rank32": "ptt5-rank32", "ptt5-full": "ptt5-full", "qwen05b": "qwen"}
            entries = load_ft_preds(name_map[model])
        preds = []
        found = 0
        for i in unique_idxs:
            src_s = sources[i].strip()
            if src_s in entries:
                preds.append(entries[src_s])
                found += 1
            else:
                preds.append("")
        all_preds[model] = preds
        print(f"  [{MODEL_DISPLAY[model]}] {found}/{len(unique_idxs)}")

    client = LLMClient(
        model=JUDGE_MODEL,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=200,
        openrouter_key=os.getenv("OPENROUTER_API_KEY"),
    )

    all_results = []
    n_total = len(unique_idxs) * len(MODEL_ORDER) * N_RUNS
    n_sanity = len(unique_idxs) * N_RUNS
    print(f"\nTotal avaliações: {n_total} (modelos) + {n_sanity} (sanity check) = {n_total + n_sanity}")
    print(f"Juiz: {JUDGE_MODEL} (temperature={JUDGE_TEMPERATURE}, {N_RUNS} runs)")

    done = 0
    for run in range(N_RUNS):
        print(f"\n--- Run {run + 1}/{N_RUNS} ---")
        for j, idx in enumerate(unique_idxs):
            original = sources[idx]
            reference = refs[idx]

            for model in MODEL_ORDER:
                prediction = all_preds[model][j]
                if not prediction:
                    continue

                prompt = EVAL_PROMPT.format(original=original, reference=reference, prediction=prediction)
                pred_hint = hashlib.md5(prediction.encode("utf-8")).hexdigest()[:12] if model not in BEST_STRATEGY else ""
                cache_key_str = judge_cache_key(original, model, run, pred_hint)
                ratings = evaluate(client, prompt, cache_key_str, run)

                if ratings:
                    all_results.append({
                        "idx": idx,
                        "run": run,
                        "model": model,
                        "model_display": MODEL_DISPLAY[model],
                        "ratings": ratings,
                        "type": "model",
                    })
                done += 1
                if done % 50 == 0:
                    print(f"  {done}/{n_total + n_sanity}")

            # Sanity check: evaluate the reference itself
            prompt = EVAL_PROMPT.format(original=original, reference=reference, prediction=reference)
            cache_key_str = judge_cache_key(original, "reference", run)
            ratings = evaluate(client, prompt, cache_key_str, run)
            if ratings:
                all_results.append({
                    "idx": idx,
                    "run": run,
                    "model": "reference",
                    "model_display": "Referência Humana",
                    "ratings": ratings,
                    "type": "sanity",
                })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "llm_judge_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResultados salvos em {out_path}")
    print(f"Total de avaliações: {len(all_results)}")

    # Summary
    from collections import defaultdict
    by_model = defaultdict(list)
    for r in all_results:
        if r["type"] == "model":
            by_model[r["model"]].append(r)

    print("\n=== RESUMO RÁPIDO ===")
    for model, entries in sorted(by_model.items()):
        dim_scores = {d: [] for d in DIMENSIONS}
        for e in entries:
            for d in DIMENSIONS:
                dim_scores[d].append(e["ratings"][d])
        line = f"  {MODEL_DISPLAY[model]}: "
        for d in DIMENSIONS:
            vals = dim_scores[d]
            if vals:
                avg = sum(vals) / len(vals)
                line += f"{d}={avg:.2f}  "
        print(line)

    sanity = [r for r in all_results if r["type"] == "sanity"]
    if sanity:
        dim_scores = {d: [] for d in DIMENSIONS}
        for e in sanity:
            for d in DIMENSIONS:
                dim_scores[d].append(e["ratings"][d])
        line = "  Ref. Humana (sanity): "
        for d in DIMENSIONS:
            vals = dim_scores[d]
            if vals:
                avg = sum(vals) / len(vals)
                line += f"{d}={avg:.2f}  "
        print(line)


if __name__ == "__main__":
    main()
