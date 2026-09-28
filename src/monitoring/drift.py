"""Embedding drift (primary signal) and categorical distribution drift (auxiliary signals).

Embedding drift: squared Maximum Mean Discrepancy with an RBF kernel (Gretton et al., 2012),
bandwidth set by the median heuristic on the pooled sample, significance by permutation test.
MMD^2 is 0 iff the two distributions are equal (for characteristic kernels), so it is a
defensible multivariate two-sample statistic for 768-d embeddings where per-dimension tests
would need heavy multiple-testing correction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class SignalResult:
    name: str
    score: float
    threshold: float
    drifted: bool
    status: str  # stable | warning | drift
    details: dict

    def to_dict(self) -> dict:
        return asdict(self)


def status_for(score: float, threshold: float, warning_ratio: float, drifted: bool) -> str:
    if drifted:
        return "drift"
    if score >= warning_ratio * threshold:
        return "warning"
    return "stable"


def _subsample(x: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if len(x) <= n:
        return x
    return x[rng.choice(len(x), n, replace=False)]


def _sq_dists(z: np.ndarray) -> np.ndarray:
    sq = (z * z).sum(1)
    d = sq[:, None] + sq[None, :] - 2 * z @ z.T
    return np.maximum(d, 0)


def _mmd2_from_kernel(k: np.ndarray, n: int) -> float:
    """Unbiased MMD^2 given pooled kernel matrix, first n rows = reference."""
    m = k.shape[0] - n
    kxx, kyy, kxy = k[:n, :n], k[n:, n:], k[:n, n:]
    sxx = (kxx.sum() - np.trace(kxx)) / (n * (n - 1))
    syy = (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
    return float(sxx + syy - 2 * kxy.mean())


def mmd_rbf(reference: np.ndarray, current: np.ndarray, permutations: int = 200,
            max_samples: int = 500, seed: int = 0) -> tuple[float, float, float]:
    """Return (mmd2, p_value, bandwidth). Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    x = _subsample(np.asarray(reference, np.float64), max_samples, rng)
    y = _subsample(np.asarray(current, np.float64), max_samples, rng)
    if len(x) < 2 or len(y) < 2:
        raise ValueError("need at least 2 samples per set")
    z = np.vstack([x, y])
    d = _sq_dists(z)
    med = np.median(d[np.triu_indices_from(d, k=1)])
    bandwidth = float(med) if med > 0 else 1.0
    k = np.exp(-d / bandwidth)
    n = len(x)
    observed = _mmd2_from_kernel(k, n)
    if permutations <= 0:
        return observed, float("nan"), bandwidth
    exceed = 0
    for _ in range(permutations):
        p = rng.permutation(len(z))
        if _mmd2_from_kernel(k[np.ix_(p, p)], n) >= observed:
            exceed += 1
    p_value = (exceed + 1) / (permutations + 1)
    return observed, p_value, bandwidth


def embedding_drift(reference: np.ndarray, current: np.ndarray, cfg: dict,
                    warning_ratio: float, seed: int = 0) -> SignalResult:
    mmd2, p, bw = mmd_rbf(reference, current, cfg["permutations"], cfg["max_samples"], seed)
    score = max(mmd2, 0.0)
    drifted = score >= cfg["threshold"] and p < cfg["p_value"]
    return SignalResult(
        "embedding", round(score, 5), cfg["threshold"], drifted,
        status_for(score, cfg["threshold"], warning_ratio, drifted),
        {"method": "mmd_rbf", "p_value": round(p, 4), "bandwidth": round(bw, 4),
         "n_reference": int(len(reference)), "n_current": int(len(current))},
    )


# ---- categorical distributions -------------------------------------------------------------
def proportions(values: list[str], categories: list[str]) -> dict[str, float]:
    n = len(values)
    if n == 0:
        return {c: 0.0 for c in categories}
    return {c: sum(v == c for v in values) / n for c in categories}


def total_variation(p: dict[str, float], q: dict[str, float]) -> float:
    keys = set(p) | set(q)
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in keys)


def categorical_drift(name: str, reference: dict[str, dict[str, float]],
                      current: dict[str, dict[str, float]], cfg: dict,
                      warning_ratio: float) -> SignalResult:
    """reference/current map a sub-signal (e.g. 'sentiment') to a proportion dict.
    Score = max total-variation distance across sub-signals."""
    per = {k: round(total_variation(reference[k], current[k]), 5) for k in reference}
    score = max(per.values()) if per else 0.0
    drifted = score >= cfg["threshold"]
    return SignalResult(name, round(score, 5), cfg["threshold"], drifted,
                        status_for(score, cfg["threshold"], warning_ratio, drifted),
                        {"method": "total_variation", "per_signal": per,
                         "reference": reference, "current": current})
