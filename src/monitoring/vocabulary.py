"""Vocabulary / n-gram change between reference and current text windows.

Supporting evidence only: a phrase being new or frequent says nothing about toxicity.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
STOPWORDS = frozenset(
    "a an the and or but is are was were be been to of in on at for with this that it its i you "
    "he she we they my your our their me him her us them so as if not no do does did have has had "
    "just very can will would could should from by about into than then there here what which who "
    "all any some more most http user".split()
)


def tokens(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN.findall(text)) if t not in STOPWORDS]


def ngram_counts(texts: list[str], n_max: int = 2) -> Counter:
    c: Counter = Counter()
    for t in texts:
        toks = tokens(t)
        for n in range(1, n_max + 1):
            c.update(" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1))
    return c


def vocabulary_report(
    reference: list[str], current: list[str], top_k: int = 25, min_count: int = 3
) -> dict:
    ref, cur = ngram_counts(reference), ngram_counts(current)
    ref_total, cur_total = max(sum(ref.values()), 1), max(sum(cur.values()), 1)
    ref_uni = {k for k in ref if " " not in k}
    cur_uni = {k: v for k, v in cur.items() if " " not in k}

    new_terms = sorted(
        ((k, v) for k, v in cur.items() if k not in ref and v >= min_count),
        key=lambda kv: (-kv[1], kv[0]),
    )[:top_k]
    shifts = []
    for k, v in cur.items():
        if v < min_count:
            continue
        p_cur = v / cur_total
        p_ref = (ref.get(k, 0) + 0.5) / ref_total  # add-half smoothing
        shifts.append((k, v, math.log2(p_cur / p_ref)))
    shifts.sort(key=lambda x: (-x[2], x[0]))
    oov = sum(v for k, v in cur_uni.items() if k not in ref_uni) / max(sum(cur_uni.values()), 1)
    jaccard = len(ref_uni & set(cur_uni)) / max(len(ref_uni | set(cur_uni)), 1)
    return {
        "top_current": [{"term": k, "count": v} for k, v in cur.most_common(top_k)],
        "new_terms": [{"term": k, "count": v} for k, v in new_terms],
        "frequency_shifts": [
            {"term": k, "count": v, "log2_ratio": round(r, 3)} for k, v, r in shifts[:top_k]
        ],
        "oov_rate": round(oov, 4),
        "unigram_jaccard": round(jaccard, 4),
    }
