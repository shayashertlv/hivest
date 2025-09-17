"""hivest/shared/risk.py"""
"""
Pure-Python risk metrics utilities.
"""
from typing import Dict, List, Tuple, Iterable, Optional


def _mean(xs: Iterable[float]) -> float:
    xs = list(x for x in xs if isinstance(x, (int, float)))
    n = len(xs)
    if n == 0:
        return 0.0
    return sum(xs) / n


def _variance(xs: Iterable[float]) -> float:
    xs = list(x for x in xs if isinstance(x, (int, float)))
    n = len(xs)
    if n <= 1:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / n  # population variance


def _covariance(xs: Iterable[float], ys: Iterable[float]) -> float:
    xs = list(xs)
    ys = list(ys)
    n = min(len(xs), len(ys))
    if n == 0:
        return 0.0
    xs = xs[:n]
    ys = ys[:n]
    mx = _mean(xs)
    my = _mean(ys)
    return sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / n


def compute_volatility(returns: List[float]) -> float:
    """Standard deviation (population) of periodic returns."""
    return _variance(returns) ** 0.5


def compute_beta(portfolio_returns: List[float], market_returns: List[float]) -> float:
    var_mkt = _variance(market_returns)
    if var_mkt == 0:
        return 0.0
    cov = _covariance(portfolio_returns, market_returns)
    return cov / var_mkt


def compute_sharpe(returns: List[float], risk_free_rate: float = 0.0) -> float:
    """
    Sharpe ratio using population std dev. Assumes returns are in same periodicity as risk_free_rate.
    If risk_free_rate is annual and returns are daily, caller should convert before calling.
    """
    excess = [r - risk_free_rate for r in returns or []]
    vol = compute_volatility(excess)
    if vol == 0:
        return 0.0
    mu = _mean(excess)
    return mu / vol

def compute_drawdown_stats(returns: List[float]) -> Dict[str, float]:
    """
    Computes max drawdown and Calmar-like ratio (cum_return / |max_drawdown|).
    Assumes periodic simple returns.
    """
    eq = [1.0]
    for r in returns or []:
        eq.append(eq[-1] * (1.0 + (r if isinstance(r, (int, float)) else 0.0)))
    peaks = []
    max_dd = 0.0
    for i, v in enumerate(eq):
        if i == 0:
            peaks.append(v)
        else:
            peaks.append(max(peaks[-1], v))
        dd = (v / peaks[-1]) - 1.0
        if dd < max_dd:
            max_dd = dd
    total = eq[-1] - 1.0 if eq else 0.0
    calmar = (total / abs(max_dd)) if max_dd < 0 else 0.0
    return {"max_drawdown": max_dd, "calmar_like": calmar}

def compute_sortino(returns: List[float], target: float = 0.0) -> float:
    """
    Sortino ratio using downside deviation relative to target (per-period).
    """
    downs = [min(0.0, (r - target)) for r in (returns or []) if isinstance(r, (int, float))]
    n = len(returns or [])
    if n == 0:
        return 0.0
    mu = _mean(returns) - target
    if not downs:
        return float('inf') if mu > 0 else 0.0
    dd = (_mean([d*d for d in downs]) ** 0.5)
    return (mu / dd) if dd else 0.0


def compute_concentration(weights: Dict[str, float], threshold: float = 0.1) -> Dict:
    """
    Computes concentration metrics:
    - hhi: Herfindahl–Hirschman Index (sum of squared weights)
    - top_weight: largest single weight
    - warnings: list of strings if any position exceeds threshold
    """
    ws = [max(0.0, float(v)) for v in (weights or {}).values()]
    total = sum(ws) or 1.0
    norm = [w / total for w in ws]
    hhi = sum(w * w for w in norm)
    top_weight = max(norm) if norm else 0.0
    eff_n = (1.0 / hhi) if hhi > 0 else 0.0
    offenders = [k for k, v in (weights or {}).items() if (max(0.0, float(v)) / total) > threshold]
    warnings = []
    if offenders:
        warnings.append(f"Positions over {threshold:.0%}: {', '.join(map(str, offenders))}")
    return {"hhi": hhi, "effective_n": eff_n, "top_weight": top_weight, "warnings": warnings}
