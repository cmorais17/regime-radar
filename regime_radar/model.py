"""
Statistical regime detection model
"""
from dataclasses import dataclass
from typing import List, Literal, Sequence

import numpy as np

PenaltySpec = float|str
ModelMode = Literal["mean", "mean-var"]

@dataclass
class Regime:
    """
    A detected regime in a piecewise-constant mean model (0-based)
    start, end are indexed inclusively
    """
    start: int
    end: int
    mean: float
    sse: float

@dataclass
class SegmentationResult:
    """
    Result of running the penalized GMS segmentation
    """
    regimes: List[Regime]
    penalty: float
    total_cost: float # sum(SSE_k + penalty) over segments

def _resolve_penalty(penalty: PenaltySpec, n: int) -> float:
    """
    Turn the penalty specified by the user into a scalar λ
    
    "bic": λ = log(n)
    "aic": λ = 2.0
    float: λ = value given
    """
    if isinstance(penalty, (int, float)):
        if penalty <= 0:
            raise ValueError("Penalty must be positive")
        return float(penalty)
    
    spec = penalty.lower()
    if spec == "bic":
        if n <= 1:
            raise ValueError("Need at least two observations for BIC penalty")
        return float(np.log(n))
    if spec == "aic":
        return 2.0
    
    raise ValueError(f"Unsupported penalty specifiation {penalty!r}. Use 'bic', 'aic', or a positive float")

def _prefix_sums(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute prefix sums of x and x^2
    
    S1[t] = sum_{i=0}^{t-1} x[i]
    S2[t] = sum_{i=0}^{t-1} x^2[i]
    """
    n = x.shape[0]
    s1 = np.zeros(n+1, dtype=float)
    s2 = np.zeros(n+1, dtype=float)
    for t in range(1, n+1):
        v = x[t-1]
        s1[t] = s1[t-1] + v
        s2[t] = s2[t-1] + v*v
    return s1, s2

def _segment_sse(prefix_sum: np.ndarray, prefix_sq: np.ndarray, s: int, t: int) -> tuple[float, int]:
    """
    SSE for the segment x[s:t]
    
    sum_x  = S1[t] - S1[s]
    sum_x2 = S2[t] - S2[s]
    n_seg  = t-s
    SSE    = sum_x2 - (sum_x^2 / n_seg)
    """
    if t<= s:
        raise ValueError("Segment end must be greater than start")
    n_seg = t-s
    sum_x = prefix_sum[t] - prefix_sum[s]
    sum_x2 = prefix_sq[t] - prefix_sq[s]
    sse = float(sum_x2 - (sum_x * sum_x) / n_seg)
    return sse, n_seg

def _segment_cost(
    s1: np.ndarray,
    s2: np.ndarray,
    s: int,
    t: int,
    mode: ModelMode,
    eps: float = 1e-12
) -> float:
    """
    Segment cost for x[s:t] for model chosen
    
    mode = "mean":
        SSE(s, t) (piecewie-constant mean, common var)
    mode = "mean-var":
        (t-s) * log(SSE(s,t) / (t-s)) (Gaussian with both mean and var)
    """
    sse, m = _segment_sse(s1, s2, s, t)

    if mode == "mean":
        return sse
    
    if mode == "mean-var":
        sse_clamped = max(sse, eps)
        return m * float(np.log(sse_clamped / m))
    
    raise ValueError(f"Unsupported model mode {mode!r}")

def detect_regimes_mean_gaussian(
    returns: Sequence[float],
    penalty: PenaltySpec = "bic",
    min_seg_len: int = 20,
    mode: ModelMode = "mean-var"
) -> SegmentationResult:
    """
    Detect regime shifts in a return series using penalized least squares
    under a Guassian piecewise-constant mean model
    
    Solving: min( sum_k [ SSE_k + λ ] )
    
    SSE_k: Sum of squared errors for segement k
    λ: Penalty param controlling # of regimes
    
    Solved using DP in O(n^2) time
    
    Params:
        returns: Sequence of log returns
        penalty: 
            "bic": λ = log(n)
            "aic": λ = 2.0
            float: pos. λ
        min_seg_len: Min allowed len of each regime
        mode: "mean" or "mean-var"
        
    Returns:
        SegmentationResult: Regimes with [start, end] + the penalty and total cost
    """
    x = np.asarray(returns, dtype=float)
    n = x.shape[0]

    if n < 2:
        raise ValueError("Need at least two observations to detect regimes")
    if min_seg_len <= 0:
        raise ValueError("min_seg_len must be positive")
    if min_seg_len > n:
        raise ValueError("min_seg_len can't exceed the number of observations")
    
    if mode not in ("mean", "mean-var"):
        raise ValueError("mode must be 'mean' or 'mean-var'")
    
    lam = _resolve_penalty(penalty, n)

    s1, s2 = _prefix_sums(x)

    # F[t] is min cost for segmenting x[0:t]
    # last_change[t] is index s that starts the last segment [s:t)
    F = np.full(n+1, np.inf, dtype=float)
    last_change = np.full(n+1, -1, dtype=int)

    F[0] = 0.0

    for t in range(min_seg_len, n+1):
        best_cost = np.inf
        best_s = -1

        s_max = t - min_seg_len
        for s in range(0, s_max+1):
            if not np.isfinite(F[s]):
                continue
            seg_cost = _segment_cost(s1, s2, s, t, mode=mode)
            candidate = F[s] + seg_cost + lam
            if candidate < best_cost:
                best_cost = candidate
                best_s = s
        
        F[t] = best_cost
        last_change[t] = best_s
    
    if not np.isfinite(F[n]) or last_change[n] < 0:
        raise ValueError("Failed to compute a valid segmentation. Try reducing min_seg_len or increasing penalty")
    
    # Backtrack to recover regimes
    regimes_rev: list[Regime] = []
    t = n
    while t > 0: 
        s = int(last_change[t])
        if s < 0:
            raise ValueError("Backtracking error in segmentation")
        seg = x[s:t]
        mean = float(seg.mean())
        sse = float(((seg - mean) ** 2).sum())
        regimes_rev.append(Regime(start=s, end=t-1, mean=mean, sse=sse))
        t = s
    
    regimes = list(reversed(regimes_rev))
    total_cost = float(F[n])

    return SegmentationResult(regimes=regimes, penalty=float(lam), total_cost=total_cost)