import re
from typing import Dict, List, Tuple

import nltk

VOWELS = set("aeiouáéíóúâêîôûàèìòùãõäëïöü")
SENTENCE_TERMINALS = re.compile(r"[.!?]+")


def count_syllables(word: str) -> int:
    word = word.lower().strip(".,!?;:()[]{}""''«»-")
    if not word:
        return 0
    prev_is_vowel = False
    groups = 0
    for ch in word:
        is_vowel = ch in VOWELS
        if is_vowel and not prev_is_vowel:
            groups += 1
        prev_is_vowel = is_vowel
    return max(groups, 1)


def count_sentences(text: str) -> int:
    spans = list(nltk.sent_tokenize(text.strip(), language="portuguese"))
    return max(len(spans), 1)


def count_words(text: str) -> list[str]:
    tokens = nltk.wordpunct_tokenize(text.lower())
    return [t for t in tokens if any(c.isalpha() for c in t)]


def flesch_reading_ease(text: str) -> float:
    sentences = count_sentences(text)
    words = count_words(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0
    syllables = sum(count_syllables(w) for w in words)
    fre = 206.835 - 1.015 * (n_words / sentences) - 84.6 * (syllables / n_words)
    return round(fre, 2)


def gunning_fog(text: str) -> float:
    sentences = count_sentences(text)
    words = count_words(text)
    n_words = len(words)
    if n_words == 0:
        return 0.0
    complex_words = sum(1 for w in words if count_syllables(w) >= 3)
    fog = 0.4 * (n_words / sentences + 100 * (complex_words / n_words))
    return round(fog, 2)


def compute_readability(text: str) -> Dict[str, float]:
    return {
        "flesch": flesch_reading_ease(text),
        "fog": gunning_fog(text),
    }


def compute_readability_metrics(
    sources: List[str],
    predictions: List[str],
) -> Dict[str, float]:
    fle_src, fle_prd = 0.0, 0.0
    fog_src, fog_prd = 0.0, 0.0
    n = len(sources)

    for src, prd in zip(sources, predictions):
        fle_src += flesch_reading_ease(src)
        fle_prd += flesch_reading_ease(prd)
        fog_src += gunning_fog(src)
        fog_prd += gunning_fog(prd)

    n = max(n, 1)
    fle_src /= n
    fle_prd /= n
    fog_src /= n
    fog_prd /= n

    return {
        "flesch_original": round(fle_src, 2),
        "flesch_prediction": round(fle_prd, 2),
        "flesch_improvement": round(fle_prd - fle_src, 2),
        "fog_original": round(fog_src, 2),
        "fog_prediction": round(fog_prd, 2),
        "fog_improvement": round(fog_prd - fog_src, 2),
    }
