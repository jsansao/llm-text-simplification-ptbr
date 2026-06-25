from typing import Dict, List, Optional

import evaluate
import numpy as np

from readability import compute_readability_metrics

_rouge = None
_bertscore = None
_sari = None


def _get_rouge():
    global _rouge
    if _rouge is None:
        _rouge = evaluate.load("rouge")
    return _rouge


def _get_bertscore():
    global _bertscore
    if _bertscore is None:
        _bertscore = evaluate.load("bertscore")
    return _bertscore


def _get_sari():
    global _sari
    if _sari is None:
        _sari = evaluate.load("sari")
    return _sari


def compute_rouge(
    predictions: List[str],
    references: List[str],
) -> Dict[str, float]:
    scorer = _get_rouge()
    result = scorer.compute(predictions=predictions, references=references)
    return {
        "rouge1": round(result["rouge1"], 4),
        "rouge2": round(result["rouge2"], 4),
        "rougeL": round(result["rougeL"], 4),
    }


def compute_bertscore(
    predictions: List[str],
    references: List[str],
    lang: str = "pt",
) -> Dict[str, float]:
    scorer = _get_bertscore()
    result = scorer.compute(
        predictions=predictions,
        references=references,
        lang=lang,
    )
    scores = result["f1"]
    return {
        "bertscore_mean": round(float(np.mean(scores)), 4),
        "bertscore_std": round(float(np.std(scores)), 4),
    }


def compute_sari(
    sources: List[str],
    predictions: List[str],
    references: List[List[str]],
) -> Dict[str, float]:
    scorer = _get_sari()
    result = scorer.compute(
        sources=sources,
        predictions=predictions,
        references=references,
    )
    return {
        "sari": round(result["sari"], 4),
    }


def _prefix_keys(d: Dict[str, float], prefix: str) -> Dict[str, float]:
    return {f"{prefix}_{k}": v for k, v in d.items()}


def compute_all_metrics(
    sources: List[str],
    predictions: List[str],
    references: List[str],
    lang: str = "pt",
) -> Dict[str, float]:
    metrics = {}

    # Predição ↔ Referência (similaridade do modelo com o humano)
    try:
        metrics.update(compute_rouge(predictions, references))
    except Exception as e:
        print(f"[metrics] ROUGE failed: {e}")

    try:
        metrics.update(compute_bertscore(predictions, references, lang=lang))
    except Exception as e:
        print(f"[metrics] BERTScore failed: {e}")

    # Original ↔ Predição (o quanto o modelo alterou o texto)
    try:
        metrics.update(_prefix_keys(compute_rouge(sources, predictions), "ori_pred"))
    except Exception as e:
        print(f"[metrics] ROUGE(ori,pred) failed: {e}")

    try:
        metrics.update(_prefix_keys(compute_bertscore(sources, predictions, lang=lang), "ori_pred"))
    except Exception as e:
        print(f"[metrics] BERTScore(ori,pred) failed: {e}")

    # Original ↔ Referência (baseline: o quanto o humano alterou)
    try:
        metrics.update(_prefix_keys(compute_rouge(sources, references), "ori_ref"))
    except Exception as e:
        print(f"[metrics] ROUGE(ori,ref) failed: {e}")

    try:
        metrics.update(_prefix_keys(compute_bertscore(sources, references, lang=lang), "ori_ref"))
    except Exception as e:
        print(f"[metrics] BERTScore(ori,ref) failed: {e}")

    try:
        metrics.update(compute_sari(sources, predictions, [[r] for r in references]))
    except Exception as e:
        print(f"[metrics] SARI failed: {e}")

    # Legibilidade: modelo vs original
    try:
        metrics.update(compute_readability_metrics(sources, predictions))
    except Exception as e:
        print(f"[metrics] Readability failed: {e}")

    # Legibilidade: referência vs original (baseline humano)
    try:
        metrics.update(_prefix_keys(compute_readability_metrics(sources, references), "ref"))
    except Exception as e:
        print(f"[metrics] Readability(ref) failed: {e}")

    return metrics
