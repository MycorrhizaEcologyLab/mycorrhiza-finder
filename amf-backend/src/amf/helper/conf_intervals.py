# AMFinder - conf_intervals.py
#
# Analytic replacement for the Monte-Carlo confidence-interval stage.
#
# The quantity the original sampler estimates (a sum of independent,
# non-identically distributed Bernoulli indicators, one per tile) is a
# Poisson-binomial. Its moments are available in closed form and its PMF is
# exact and cheap whenever the support is short. Nothing here changes the
# statistical model: tiles are still assumed independent and the calibrated
# probabilities are still taken at face value.


"""
Analytic confidence intervals for AMFinder tile metrics.

Functions
------------

:function background_class_index: index of the Background class for the
    active colonisation type.
:function auto_background_mask: locates intensity-thresholded tiles.
:function tile_probabilities: row-normalised probability matrix.
:function add_conf_intervals: analytic replacement for the Monte-Carlo
    confidence-interval entry point in predict.py.
"""

from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import amf.helper.config as AmfConfig

# Tail mass per side. 0.02275 reproduces the coverage of the original
# "mean +/- 2 * stddev" bounds (95.45% central). Use 0.025 for a clean 95%.
DEFAULT_ALPHA = 0.02275

# Use the exact truncated PMF when a class's support fits in this many counts.
# Cost is O(n_tiles * max_support), so this caps it at roughly 0.25 s.
EXACT_COUNT_MAX_SUPPORT = 500

# Tiles below this probability are dropped from the exact recursion. Total
# discarded mass is bounded by n_tiles * this, i.e. ~3e-5 at 3e4 tiles.
NEGLIGIBLE = 1e-9

NORM = NormalDist()


def config_get(key: str, default: Any = None) -> Any:
    """
    AmfConfig.get() that tolerates a missing key, for new settings.
    """
    try:
        value = AmfConfig.get(key)
    except Exception:
        return default
    return default if value is None else value


# --------------------------------------------------------------------------
# background tiles
# --------------------------------------------------------------------------


def background_class_index() -> int:
    """
    Index of the Background class in AmfConfig.get("header").

    'am' order:  AMColonised, Uncolonised, Background, Unreadable, DSE, Hybrid
    'erm' order: BlueCoils, BrownCoils, TypeTwo, Uncolonised, Background,
                 MainRoot, Unreadable, DSE, HybridErm, HybridDse
    """
    return 2 if AmfConfig.get("colonisation_type") == "am" else 4


def auto_background_mask(table: pd.DataFrame) -> pd.Series:
    """
    True for tiles auto-classified as background by intensity thresholding
    or inferred as background with probability 1.0.
    """
    header = AmfConfig.get("header")
    return table[header[background_class_index()]] >= 1.0


# --------------------------------------------------------------------------
# tile probabilities  (replaces bootstrap_distribution as the input object)
# --------------------------------------------------------------------------


def tile_probabilities(table: pd.DataFrame) -> NDArray[np.float64]:
    """
    Row-normalised per-tile class probabilities, shape (n_tiles, n_classes).

    Normalisation is the same safety measure the original sample_class applied,
    done once here instead of once per draw per tile.
    """
    probs = table.loc[:, AmfConfig.get("header")].to_numpy(dtype=np.float64)
    totals = probs.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("at least one tile has zero total probability mass")
    return probs / totals


# --------------------------------------------------------------------------
# exact moments
# --------------------------------------------------------------------------


def count_moments(
    probs: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Exact mean, standard deviation and skewness of every class count.

    For N_k = sum_i Bernoulli(p_ik), with no approximation:
        E[N_k]   = sum_i p_ik
        Var[N_k] = sum_i p_ik (1 - p_ik)
        mu3[N_k] = sum_i p_ik (1 - p_ik) (1 - 2 p_ik)
    """
    mean = probs.sum(axis=0)
    var = (probs * (1.0 - probs)).sum(axis=0)
    mu3 = (probs * (1.0 - probs) * (1.0 - 2.0 * probs)).sum(axis=0)
    sd = np.sqrt(var)
    skew = np.divide(mu3, sd**3, out=np.zeros_like(mu3), where=sd > 0.0)
    return mean, sd, skew


def cornish_fisher(
    mean: float, sd: float, skew: float, alpha: float
) -> tuple[float, float]:
    """Normal quantiles with a first-order skewness correction."""
    bounds = []
    for tail in (alpha, 1.0 - alpha):
        z = NORM.inv_cdf(tail)
        bounds.append(mean + sd * (z + (z * z - 1.0) * skew / 6.0))
    return bounds[0], bounds[1]


# --------------------------------------------------------------------------
# exact truncated Poisson-binomial PMF  (rare classes)
# --------------------------------------------------------------------------


def truncated_pb_pmf(q: NDArray[np.float64], max_count: int) -> NDArray[np.float64]:
    """
    Exact PMF coefficients 0..max_count of sum_i Bernoulli(q_i).

    Truncation is safe because counts only increase: mass pushed past
    max_count can never re-enter the retained window. Every update is a
    convex combination of non-negative terms, so there is no subtractive
    cancellation and round-off accumulates as O(n * eps).
    """
    q = np.asarray(q, dtype=np.float64)
    n_certain = int(np.sum(q >= 1.0))
    q = q[(q > NEGLIGIBLE) & (q < 1.0)]

    width = max(max_count - n_certain, 1)
    pmf = np.zeros(width + 1)
    pmf[0] = 1.0
    for qi in q:
        pmf[1:] = pmf[1:] * (1.0 - qi) + pmf[:-1] * qi
        pmf[0] *= 1.0 - qi

    if n_certain:
        pmf = np.concatenate([np.zeros(n_certain), pmf])
    return pmf


def class_count_bounds(
    q: NDArray[np.float64], mean: float, sd: float, skew: float, alpha: float
) -> tuple[float, float]:
    """
    Central (1 - 2*alpha) bounds on a class count.
    """
    upper_support = mean + 8.0 * sd
    if upper_support <= EXACT_COUNT_MAX_SUPPORT:
        pmf = truncated_pb_pmf(q, int(np.ceil(upper_support)) + 1)
        cdf = np.cumsum(pmf)
        lower = float(np.searchsorted(cdf, alpha, side="left"))
        upper = float(np.searchsorted(cdf, 1.0 - alpha, side="left"))
        return lower, upper

    lower, upper = cornish_fisher(mean, sd, skew, alpha)
    return max(lower, 0.0), upper


# --------------------------------------------------------------------------
# class index sets
#
# Transcribed from prepare_metrics(); the numerator of each ratio and the
# total_root_tiles denominator, expressed as class indices.
# --------------------------------------------------------------------------

AM_DENOM = (0, 1, 4, 5)  # all classes minus Background(2), Unreadable(3)
AM_NUMER = {
    "am_colonised_percentage": (0,),
    "dse_colonised_percentage": (4,),
    "total_colonised_percentage": (0, 4),
}

# all classes minus Background(4), MainRoot(5), Unreadable(6)
ERM_DENOM = (0, 1, 2, 3, 7, 8, 9)
ERM_NUMER = {
    "BlueCoils_colonised_percentage": (0,),
    "BrownCoils_colonised_percentage": (1,),
    "TypeTwo_colonised_percentage": (2,),
    "dse_colonised_percentage": (7, 9),
    "total_colonised_percentage": (0, 1, 2, 8),
}


def index_sets(
    metric: str, include_hybrid: bool
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if AmfConfig.get("colonisation_type") == "am":
        if metric not in AM_NUMER:
            raise ValueError(
                "Not a possible metric. Choose either am_colonised_percentage, "
                "dse_colonised_percentage or total_colonised_percentage"
            )
        # Hybrid(5) is always in the denominator, in the numerator only when
        # include_hybrid is set -- same as the original boolean multiply.
        return AM_NUMER[metric] + ((5,) if include_hybrid else ()), AM_DENOM

    if metric not in ERM_NUMER:
        raise ValueError(
            "Not a possible metric. Choose either BlueCoils_colonised_percentage, "
            "BrownCoils_colonised_percentage, TypeTwo_colonised_percentage, "
            "dse_colonised_percentage or total_colonised_percentage"
        )
    return ERM_NUMER[metric], ERM_DENOM  # include_hybrid unused, as in the original


# --------------------------------------------------------------------------
# ratio metrics
# --------------------------------------------------------------------------


def ratio_delta(
    a: NDArray[np.float64], b: NDArray[np.float64], alpha: float
) -> tuple[float, float, float]:
    """
    Delta method. Numerator classes must be a subset of denominator classes.

    With A = sum_i Bernoulli(a_i) and B = sum_i Bernoulli(b_i) on the same
    tiles, Cov(A, B) = sum_i (a_i - a_i b_i) because U is a subset of D.
    """
    mu_a, mu_b = a.sum(), b.sum()
    var_a = float((a * (1.0 - a)).sum())
    var_b = float((b * (1.0 - b)).sum())
    cov = float((a * (1.0 - b)).sum())

    ratio = float(mu_a / mu_b)
    var_r = (var_a - 2.0 * ratio * cov + ratio * ratio * var_b) / (mu_b * mu_b)
    sd = float(np.sqrt(max(var_r, 0.0)))
    z = NORM.inv_cdf(1.0 - alpha)
    return ratio, max(ratio - z * sd, 0.0), min(ratio + z * sd, 1.0)


def get_relative_conf_intervals(
    probs: NDArray[np.float64],
    include_hybrid: bool,
    metric: str,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, float]:
    """
    Analytic counterpart of predict.get_relative_conf_intervals.
    """
    numer_idx, denom_idx = index_sets(metric, include_hybrid)
    if not set(numer_idx).issubset(denom_idx):
        raise ValueError(
            f"numerator classes for {metric} are not a subset of denominator"
        )

    a = probs[:, list(numer_idx)].sum(axis=1)
    b = probs[:, list(denom_idx)].sum(axis=1)

    keep = b > 0.0  # tiles that can never land in the denominator are inert
    a, b = a[keep], b[keep]

    if a.size == 0 or b.sum() <= 0.0:
        mean = lower = upper = float("nan")
    else:
        mean, lower, upper = ratio_delta(a, b, alpha)

    res_dict = {}
    res_dict[metric + "_Mean"] = mean * 100
    res_dict[metric + "_LC"] = lower * 100
    res_dict[metric + "_UC"] = upper * 100

    return res_dict


# --------------------------------------------------------------------------
# entry point  (same contract as predict.add_conf_intervals)
# --------------------------------------------------------------------------


def add_conf_intervals(
    probs: NDArray[np.float64],
    metrics_dict: dict[str, Any],
    include_hybrid: bool,
    background_count: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """
    Analytic counterpart of predict.add_conf_intervals.

    :param probs: row-normalised tile probabilities, background tiles removed.
    :param background_count: number of auto-classified background tiles that
        were removed. Added back as a constant to the Background class bounds,
        which is exact: a tile with p = 1 shifts the distribution by one and
        adds no variance.
    """
    headers = AmfConfig.get("header")
    dict_keys = metrics_dict.keys()
    mean, sd, skew = count_moments(probs)
    res = {}

    for header in dict_keys:
        if header in headers:
            i = headers.index(header)
            lower, upper = class_count_bounds(
                probs[:, i], float(mean[i]), float(sd[i]), float(skew[i]), alpha
            )
            res[header + "_Mean"] = float(mean[i])
            res[header + "_LowerConfidence"] = lower
            res[header + "_UpperConfidence"] = upper

        elif "percentage" in header:
            temp = get_relative_conf_intervals(probs, include_hybrid, header, alpha)
            res.update(temp)

        else:
            continue

    apply_background_offset(res, background_count)
    metrics_dict.update(res)

    return metrics_dict


def apply_background_offset(
    res: dict[str, float], background_count: int
) -> dict[str, float]:
    """
    Add removed auto-background tiles back to the Background class bounds.

    Keeps <Background>_Mean comparable with the raw <Background> tile count.
    Set background_count to 0 at the call site to report inference-only
    background instead.
    """
    if background_count <= 0:
        return res
    name = AmfConfig.get("header")[background_class_index()]
    for suffix in ("_Mean", "_LowerConfidence", "_UpperConfidence"):
        if name + suffix in res:
            res[name + suffix] += background_count
    return res


def compare_intervals(
    analytic: dict[str, float], sampled: dict[str, float]
) -> tuple[float, float]:
    """
    Largest absolute disagreement, split by unit.

    :return: (max difference in percentage points, max difference in tiles).
    """
    pct, counts = [0.0], [0.0]
    for key, value in analytic.items():
        if key not in sampled:
            continue
        try:
            diff = abs(float(value) - float(sampled[key]))
        except (TypeError, ValueError):
            continue
        if np.isnan(diff):
            continue
        (pct if "percentage" in key else counts).append(diff)
    return max(pct), max(counts)
