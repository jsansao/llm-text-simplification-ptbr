import hashlib
import json
import os
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import load_pairs
from llm_client import LLMClient

CACHE_DIR = Path(__file__).resolve().parent.parent / "cache2_nemotron_test"
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

JUDGE_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"
JUDGE_TEMPERATURE = 0.3
N_RUNS = 1
MAX_WORKERS = 8
MAX_SAMPLES = 20

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


def build_cache_index() -> dict[tuple[str, str], dict[str, str]]:
    cache_dir = Path(__file__).resolve().parent.parent / "cache"
    index = defaultdict(lambda: defaultdict(dict))
    for fn in os.listdir(cache_dir):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(cache_dir, fn)) as f:
            d = json.load(f)
        model = d.get("model")
        strategy = d.get("strategy")
        if model and strategy:
            src = d.get("original", "").strip()
            pred = d.get("prediction", d.get("raw_output", "")).strip()
            if src and pred:
                index[model][strategy][src] = pred
    return index


def load_ft_preds(model_name: str) -> dict[str, str]:
    name_map = {"ptt5-rank8": "ptt5", "ptt5-rank32": "ptt5-rank32", "ptt5-full": "ptt5-full", "qwen05b": "qwen"}
    path = RESULTS_DIR / f"ft-{name_map[model_name]}" / "predictions.json"
    with open(path) as f:
        data = json.load(f)
    return {src.strip(): pred.strip() for src, pred in zip(data["sources"], data["predictions"])}


def judge_cache_key(original: str, model: str, run: int) -> str:
    raw = f"llm_judge||{JUDGE_MODEL}||{JUDGE_TEMPERATURE}||run{run}||{model}||{original}"
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


_thread_local = threading.local()


def _get_client():
    if not hasattr(_thread_local, "client"):
        _thread_local.client = LLMClient(
            model=JUDGE_MODEL,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=200,
            openrouter_key=os.getenv("OPENROUTER_API_KEY"),
            rpm=max(10, 200 // MAX_WORKERS),
        )
    return _thread_local.client


def _get_cached(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f).get("ratings")
    return None


def _write_cache(key: str, prompt: str, response: str, ratings: dict):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump({
            "prompt": prompt,
            "response": response,
            "ratings": ratings,
            "timestamp": time.time(),
        }, f, ensure_ascii=False)


def process_task(task: tuple) -> dict | None:
    type_, run, idx, model, model_display, prompt, cache_key = task

    cached_ratings = _get_cached(cache_key)
    if cached_ratings is not None:
        return {
            "idx": idx,
            "run": run,
            "model": model,
            "model_display": model_display,
            "ratings": cached_ratings,
            "type": type_,
        }

    client = _get_client()
    response = client.generate(prompt, strategy="cot")
    if not response:
        print(f"  [run {run}] Empty response for {model_display}", flush=True)
        return None

    ratings = parse_ratings(response)
    if ratings is None:
        print(f"  [run {run}] Failed to parse for {model_display}: {response[:200]}", flush=True)
        return None

    _write_cache(cache_key, prompt, response, ratings)
    return {
        "idx": idx,
        "run": run,
        "model": model,
        "model_display": model_display,
        "ratings": ratings,
        "type": type_,
    }


def build_tasks(sources, refs, unique_idxs, all_preds):
    tasks = []
    for run in range(N_RUNS):
        for j, idx in enumerate(unique_idxs):
            original = sources[idx]
            reference = refs[idx]

            for model in MODEL_ORDER:
                prediction = all_preds[model][j]
                if not prediction:
                    continue
                prompt = EVAL_PROMPT.format(
                    original=original, reference=reference, prediction=prediction
                )
                cache_key = judge_cache_key(original, model, run)
                tasks.append(("model", run, idx, model, MODEL_DISPLAY[model], prompt, cache_key))

            prompt = EVAL_PROMPT.format(
                original=original, reference=reference, prediction=reference
            )
            cache_key = judge_cache_key(original, "reference", run)
            tasks.append(("sanity", run, idx, "reference", "Referência Humana", prompt, cache_key))
    return tasks


def main():
    print("=" * 60)
    print(f"TESTE: Nemotron 3 Ultra 550B como juiz")
    print(f"  Modelo: {JUDGE_MODEL}")
    print(f"  Temperature: {JUDGE_TEMPERATURE}")
    print(f"  Runs: {N_RUNS}")
    print(f"  Max samples: {MAX_SAMPLES}")
    print(f"  Cache dir: {CACHE_DIR}")
    print("=" * 60)

    print("\nBuilding cache index...")
    cache_index = build_cache_index()
    print(f"  Indexed {sum(len(v) for m in cache_index.values() for v in m.values())} entries across {len(cache_index)} models")

    print("Loading dataset...")
    df = load_pairs(max_samples=MAX_SAMPLES, random_state=42)
    sources = df["original"].tolist()
    refs = df["simplified"].tolist()

    seen = set()
    unique_idxs = []
    for i, src in enumerate(sources):
        s = src.strip()
        if s not in seen:
            seen.add(s)
            unique_idxs.append(i)
    print(f"  {len(unique_idxs)} unique samples")

    all_preds = {}
    for model in MODEL_ORDER:
        if model in BEST_STRATEGY:
            strat = BEST_STRATEGY[model]
            entries = cache_index.get(model, {}).get(strat, {})
        else:
            entries = load_ft_preds(model)
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

    tasks = build_tasks(sources, refs, unique_idxs, all_preds)
    total = len(tasks)

    remaining = []
    for t in tasks:
        cache_key = t[6]
        if not (CACHE_DIR / f"{cache_key}.json").exists():
            remaining.append(t)
    print(f"\nTotal evaluations: {total}")
    print(f"  Already cached: {total - len(remaining)}")
    print(f"  Remaining: {len(remaining)}")

    t_start = time.time()

    if remaining:
        done_lock = threading.Lock()
        done_count = 0
        fail_count = 0
        all_results = []

        def task_wrapper(t):
            nonlocal fail_count
            result = process_task(t)
            with done_lock:
                nonlocal done_count
                done_count += 1
                if done_count % 20 == 0 or done_count == len(remaining):
                    elapsed = time.time() - t_start
                    rate = done_count / elapsed if elapsed > 0 else 0
                    print(f"  {done_count}/{len(remaining)} ({rate:.1f} calls/s)", flush=True)
                if result is None:
                    fail_count += 1
                else:
                    all_results.append(result)
            return result

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(task_wrapper, t) for t in remaining]
            for _ in as_completed(futures):
                pass

        t_elapsed = time.time() - t_start
        print(f"\nDone in {t_elapsed:.1f}s. Success: {len(all_results)}, Failed: {fail_count}")

    all_results = []
    for t in tasks:
        type_, run, idx, model, model_display, prompt, cache_key = t
        cached_ratings = _get_cached(cache_key)
        if cached_ratings:
            all_results.append({
                "idx": idx,
                "run": run,
                "model": model,
                "model_display": model_display,
                "ratings": cached_ratings,
                "type": type_,
            })

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "llm_judge_results_nemotron_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {out_path}")
    print(f"Total evaluations: {len(all_results)}")

    by_model = defaultdict(list)
    for r in all_results:
        if r["type"] == "model":
            by_model[r["model"]].append(r)

    parse_success = {}
    print("\n=== NEMOTRON 3 ULTRA 550B - RESULTADOS ===")
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
        parse_success[model] = sum(1 for e in entries) / len(entries) * 100 if entries else 0

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

    print(f"\n=== COMPARAÇÃO COM GPT-4o MINI ===")
    gpt_path = RESULTS_DIR / "llm_judge_results_gpt4omini.json"
    if gpt_path.exists():
        with open(gpt_path) as f:
            gpt_results = json.load(f)

        by_model_gpt = defaultdict(list)
        for r in gpt_results:
            if r["type"] == "model" and r["run"] == 0:
                by_model_gpt[r["model"]].append(r)

        for model in MODEL_ORDER:
            ds_entries = by_model.get(model, [])
            gpt_entries = by_model_gpt.get(model, [])
            if not ds_entries or not gpt_entries:
                continue
            line = f"  {MODEL_DISPLAY[model]}:\n"
            for d in DIMENSIONS:
                ds_vals = [e["ratings"][d] for e in ds_entries]
                gpt_vals = [e["ratings"][d] for e in gpt_entries]
                ds_avg = sum(ds_vals) / len(ds_vals)
                gpt_avg = sum(gpt_vals) / len(gpt_vals)
                diff = ds_avg - gpt_avg
                line += f"    {d}: Nem={ds_avg:.2f}  GPT={gpt_avg:.2f}  diff={diff:+.2f}\n"
            print(line)

    print("=" * 60)


if __name__ == "__main__":
    main()
