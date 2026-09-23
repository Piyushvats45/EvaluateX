from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

_ROUGE_BACKEND = None
try:
    from rouge_score import rouge_scorer as _rs

    _rouge_scorer_obj = _rs.RougeScorer(["rougeL"], use_stemmer=True)
    _ROUGE_BACKEND = "rouge_score"
except ImportError:
    _rouge_scorer_obj = None

_BERTSCORE_AVAILABLE = False
try:
    import bert_score  # noqa: F401

    _BERTSCORE_AVAILABLE = True

    try:
        from transformers.utils import logging as _hf_logging

        _hf_logging.set_verbosity_error()
    except Exception:
        pass
except ImportError:
    _BERTSCORE_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Longest common subsequence length (pure Python, O(len(a)*len(b)))."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]

def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 score in [0, 1]. Uses `rouge-score` if installed, else a
    pure-Python LCS fallback that computes the same underlying statistic."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0

    if _ROUGE_BACKEND == "rouge_score":
        scores = _rouge_scorer_obj.score(reference, hypothesis)
        return float(scores["rougeL"].fmeasure)

    # --- pure-python fallback ---
    ref_tokens = _tokenize(reference)
    hyp_tokens = _tokenize(hypothesis)
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = _lcs_length(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hyp_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_BERT_SCORER_CACHE: dict[str, "object"] = {}


def _get_bert_scorer(lang: str):
    if lang not in _BERT_SCORER_CACHE:
        from bert_score import BERTScorer

        _BERT_SCORER_CACHE[lang] = BERTScorer(lang=lang, rescale_with_baseline=False)
    return _BERT_SCORER_CACHE[lang]

def semantic_similarity(
    reference: str, hypothesis: str, lang: str = "en"
) -> tuple[float, str]:
    """Returns (score in [0,1], backend_used).

    Tries real BERTScore first (still free — runs a local open-weight model,
    no API key, no per-call cost). The backbone model is loaded once and
    cached (see _get_bert_scorer) rather than reloaded on every call. Falls
    back to TF-IDF cosine similarity, which needs no model download at all
    and is genuinely $0 / offline.
    """
    if not reference.strip() or not hypothesis.strip():
        return 0.0, "none"

    if _BERTSCORE_AVAILABLE:
        try:
            scorer = _get_bert_scorer(lang)
            _, _, f1 = scorer.score([hypothesis], [reference])
            return float(f1.mean()), "bert-score"
        except Exception:
            pass 

    if _SKLEARN_AVAILABLE:
        vec = TfidfVectorizer().fit([reference, hypothesis])
        matrix = vec.transform([reference, hypothesis])
        sim = cosine_similarity(matrix[0], matrix[1])[0][0]
        return float(sim), "tfidf-cosine"

    ref_set, hyp_set = set(_tokenize(reference)), set(_tokenize(hypothesis))
    if not ref_set or not hyp_set:
        return 0.0, "jaccard"
    jaccard = len(ref_set & hyp_set) / len(ref_set | hyp_set)
    return jaccard, "jaccard"


def distinct_n(text: str, n: int = 2) -> float:
    """Fraction of unique n-grams — a proxy for repetitiveness/degeneration."""
    tokens = _tokenize(text)
    if len(tokens) < n:
        return 1.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 1.0
    return len(set(ngrams)) / len(ngrams)


def response_length(text: str) -> int:
    return len(_tokenize(text))


# Composite scoring
@dataclass
class MetricResult:
    rouge_l: float
    semantic_sim: float
    semantic_backend: str
    distinct_2: float
    length: int
    composite: float


def score_response(
    reference: Optional[str],
    hypothesis: str,
    weights: Optional[dict] = None,
) -> MetricResult:
    """Compute the full structured metric suite for one (reference, hypothesis)
    pair. If `reference` is None (no gold answer, e.g. open-ended generation),
    reference-based metrics are set to 0.0 and composite falls back to
    diversity/length heuristics only.
    """
    weights = weights or {"rouge_l": 0.4, "semantic_sim": 0.4, "distinct_2": 0.2}

    if reference:
        r_l = rouge_l(reference, hypothesis)
        sem, backend = semantic_similarity(reference, hypothesis)
    else:
        r_l, sem, backend = 0.0, 0.0, "n/a"

    d2 = distinct_n(hypothesis, 2)
    length = response_length(hypothesis)

    if reference:
        composite = (
            weights["rouge_l"] * r_l
            + weights["semantic_sim"] * sem
            + weights["distinct_2"] * d2
        )
    else:
        composite = d2  

    return MetricResult(
        rouge_l=r_l,
        semantic_sim=sem,
        semantic_backend=backend,
        distinct_2=d2,
        length=length,
        composite=round(composite, 4),
    )


def metric_human_agreement(auto_scores: list[float], human_scores: list[float]) -> dict:
    """Pearson + Spearman-style rank agreement between automated metrics and
    human ratings — the same "metric-human agreement" analysis used to
    validate whether automated scoring is a trustworthy proxy for human
    preference judgments.
    """
    import numpy as np

    if len(auto_scores) != len(human_scores) or len(auto_scores) < 2:
        return {"pearson": None, "spearman": None, "n": len(auto_scores)}

    a = np.array(auto_scores, dtype=float)
    h = np.array(human_scores, dtype=float)

    pearson = float(np.corrcoef(a, h)[0, 1]) if a.std() > 0 and h.std() > 0 else None

    def _rank(x):
        return np.argsort(np.argsort(x))

    if a.std() > 0 and h.std() > 0:
        ra, rh = _rank(a), _rank(h)
        spearman = float(np.corrcoef(ra, rh)[0, 1])
    else:
        spearman = None

    return {"pearson": pearson, "spearman": spearman, "n": len(auto_scores)}
