"""hivest/shared/performance.py"""
"""
Performance and attribution helpers (pure Python, dependency-light).
Note: To avoid circular imports, holdings are generic dicts with keys: symbol, weight, sector (optional).
"""
from typing import List, Dict, Tuple, Optional


def _safe_list(xs):
    return [x for x in (xs or []) if isinstance(x, (int, float))]


def _prod(xs: List[float]) -> float:
    p = 1.0
    for x in xs:
        p *= (1.0 + x)
    return p


def cumulative_return(returns: List[float]) -> float:
    returns = _safe_list(returns)
    if not returns:
        return 0.0
    return _prod(returns) - 1.0


def annualized_return(returns: List[float], periods_per_year: float) -> float:
    returns = _safe_list(returns)
    n = len(returns)
    if n == 0 or periods_per_year <= 0:
        return 0.0
    import math
    try:
        g = sum(math.log1p(r) for r in returns)  # may ValueError if any (1+r) <= 0
        years = n / periods_per_year
        return math.expm1(g / years)
    except ValueError:
        total = _prod(returns)  # falls back to geometric if valid
        years = n / periods_per_year
        if total <= 0 or years <= 0:
            return 0.0
        return (total ** (1.0 / years) - 1.0)


def benchmark_comparison(portfolio_returns: List[float], benchmarks: Dict[str, List[float]]) -> Dict:
    """
    Computes cumulative returns for portfolio and each benchmark, and relative difference (portfolio - benchmark).
    """
    port_cum = cumulative_return(portfolio_returns)
    out = {"portfolio_cum": port_cum, "benchmarks": {}}
    for name, rets in (benchmarks or {}).items():
        b_cum = cumulative_return(rets)
        out["benchmarks"][name] = {
            "cum_return": b_cum,
            "relative_vs_portfolio": port_cum - b_cum,
        }
    return out


def attribution_by_position(holdings: List[Dict], position_returns: Optional[Dict[str, List[float]]] = None) -> Dict:
    """
    Returns contribution per symbol using weight * cumulative(symbol_return) if available,
    otherwise uses weights as a proxy (not ideal, but a clear placeholder when per-symbol returns missing).
    Output: {
      'by_symbol': List[Tuple[symbol, contribution]], sorted desc,
      'winners': top 5, 'losers': bottom 5
    }
    """
    contributions: List[Tuple[str, float]] = []
    weights = { (h.get("symbol") or "").upper(): float(h.get("weight") or 0.0) for h in (holdings or []) }

    if position_returns:
        for sym, w in weights.items():
            sym_rets = position_returns.get(sym)
            contrib = w * cumulative_return(sym_rets) if sym_rets else 0.0
            contributions.append((sym, contrib))
    else:
        # Fallback: weight as proxy for contribution
        contributions = list(weights.items())

    contributions.sort(key=lambda t: t[1], reverse=True)
    winners = contributions[:5]
    losers = sorted(contributions, key=lambda t: t[1])[:5]
    return {"by_symbol": contributions, "winners": winners, "losers": losers}


def attribution_by_sector(holdings: List[Dict], position_returns: Optional[Dict[str, List[float]]] = None) -> Dict:
    """
    Aggregates contributions per sector. If per-symbol returns available, uses weight * cum(symbol_return),
    else uses weight as proxy.
    Output: {'by_sector': List[Tuple[sector, contribution]], 'top': top5, 'bottom': bottom5}
    """
    sector_contrib: Dict[str, float] = {}
    for h in holdings or []:
        sym = (h.get("symbol") or "").upper()
        sector = h.get("sector") or "Unknown"
        w = float(h.get("weight") or 0.0)
        if position_returns and sym in position_returns:
            contrib = w * cumulative_return(position_returns[sym])
        else:
            contrib = w
        sector_contrib[sector] = sector_contrib.get(sector, 0.0) + contrib

    items = sorted(sector_contrib.items(), key=lambda t: t[1], reverse=True)
    top = items[:5]
    bottom = sorted(items, key=lambda t: t[1])[:5]
    return {"by_sector": items, "top": top, "bottom": bottom}
