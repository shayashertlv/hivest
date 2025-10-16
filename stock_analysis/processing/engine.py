"""hivest/stock_analysis/processing/engine.py"""
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple

from .models import StockInput, AnalysisOptions, ComputedMetrics, StockAnalysisContext
from ...shared.market import build_per_symbol_returns, infer_yahoo_range, fetch_yahoo_chart, compute_simple_returns
from ...shared.performance import cumulative_return
from ...shared.risk import compute_volatility, compute_beta, compute_sharpe, compute_drawdown_stats, compute_sortino
from ...shared.news import fetch_news_api
from ...shared.events import next_earnings_date
from ...shared.fundamentals import get_fundamentals


# --- Simple technical helpers (kept local to avoid adding dependencies) ---
def _sma(values: List[float], window: int) -> Optional[float]:
    if not isinstance(values, list) or len(values) < window or window <= 0:
        return None
    try:
        return sum(values[-window:]) / float(window)
    except Exception:
        return None


def _rsi_from_prices(closes: List[float], window: int = 14) -> Optional[float]:
    try:
        if not closes or len(closes) < window + 1:
            return None
        gains: List[float] = []
        losses: List[float] = []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i-1]
            gains.append(max(0.0, diff))
            losses.append(max(0.0, -diff))
        # Use last `window` periods average
        avg_gain = sum(gains[-window:]) / window
        avg_loss = sum(losses[-window:]) / window
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))
    except Exception:
        return None


def _pct_from_extrema(closes: List[float], lookback_days: int = 252) -> Tuple[Optional[float], Optional[float]]:
    """Return (pct_from_52w_high, pct_from_52w_low) using available closes and ~1y lookback."""
    try:
        if not closes:
            return None, None
        window = min(len(closes), max(1, lookback_days))
        sub = closes[-window:]
        hi = max(sub)
        lo = min(sub)
        last = sub[-1]
        pct_from_high = (last / hi - 1.0) if hi else None
        pct_from_low = (last / lo - 1.0) if lo else None
        return pct_from_high, pct_from_low
    except Exception:
        return None, None


def analyze_stock(si: StockInput, options: Optional[AnalysisOptions] = None) -> Dict[str, Any]:
    """
    Compute a compact analysis context for a single symbol, mirroring the portfolio flow:
    - fill aligned return series for symbol and benchmarks (SPY, QQQ)
    - compute core performance/risk stats
    - compute light technicals (RSI, SMAs, distances to 52w extrema)
    - fetch fundamentals and (optionally) news and upcoming events
    Returns a dict with keys used by the API and prompt builder.
    """
    options = options or AnalysisOptions()
    symbol = (si.symbol or "").upper()

    # --- 1) Returns for symbol and benchmarks ---
    rets_map = build_per_symbol_returns([symbol, 'SPY', 'QQQ'], si.timeframe_label)
    sym_rets = rets_map.get(symbol) or []
    spy_rets = rets_map.get('SPY') or []
    qqq_rets = rets_map.get('QQQ') or []

    # If build_per_symbol_returns failed for symbol, try a direct fetch to compute technicals later
    yf_range = infer_yahoo_range(si.timeframe_label)
    ds, closes = fetch_yahoo_chart(symbol, yf_range=yf_range, interval='1d')
    if not sym_rets and closes:
        sym_rets = compute_simple_returns(closes)

    # --- 2) Core metrics ---
    has_rets = bool(sym_rets)
    cum_ret = cumulative_return(sym_rets) if has_rets else None
    vol = compute_volatility(sym_rets) if has_rets else None
    mkt_rets = spy_rets or si.market_returns or []
    beta = compute_beta(sym_rets, mkt_rets) if has_rets else None
    sharpe = compute_sharpe(sym_rets, si.risk_free_rate) if has_rets else None
    sortino = compute_sortino(sym_rets) if has_rets else None
    dd_stats = compute_drawdown_stats(sym_rets) if has_rets else {}

    # --- 3) Benchmarks block ---
    benchmarks: Dict[str, Dict[str, float]] = {}
    if spy_rets:
        from ...shared.performance import cumulative_return as _cr
        b = _cr(spy_rets)
        benchmarks['SPY'] = {"cum_return": b, "relative_vs_asset": cum_ret - b}
    if qqq_rets:
        from ...shared.performance import cumulative_return as _cr
        b = _cr(qqq_rets)
        benchmarks['QQQ'] = {"cum_return": b, "relative_vs_asset": cum_ret - b}

    # --- 4) Technicals ---
    rsi = _rsi_from_prices(closes) if closes else None
    sma20 = _sma(closes, 20) if closes else None
    sma50 = _sma(closes, 50) if closes else None
    sma200 = _sma(closes, 200) if closes else None
    pct_hi, pct_lo = _pct_from_extrema(closes, lookback_days=252)

    # --- 5) Fundamentals and instrument type ---
    inst_type, fund, etf_prof = get_fundamentals(symbol)

    # --- 6) News ---
    news_items: List[Dict[str, Any]] = []
    if options.include_news:
        news_items = fetch_news_api([symbol], limit=max(1, options.news_limit))

    # --- 7) Upcoming events ---
    upcoming: List[Dict[str, Any]] = []
    if options.include_events:
        nxt = next_earnings_date(symbol)
        if nxt:
            upcoming.append({"symbol": symbol, "type": "earnings", "date": nxt, "note": "Next earnings date (FMP)"})

    last_price = closes[-1] if closes else None

    cm = ComputedMetrics(
        cum_return=cum_ret,
        volatility=vol,
        beta=beta,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=(dd_stats.get("max_drawdown") if dd_stats else None),
        benchmarks=benchmarks,
        rsi=rsi,
        sma20=sma20,
        sma50=sma50,
        sma200=sma200,
        pct_from_52w_high=pct_hi,
        pct_from_52w_low=pct_lo,
        last_price=last_price,
        instrument_type=inst_type,
        fundamentals=fund or {},
        etf_profile=etf_prof or {},
    )

    ctx: Dict[str, Any] = {
        "stock_input": si,
        "computed_metrics": cm,
        "news_items": news_items,
        "upcoming_events": upcoming,
    }

    # Debug logging: context snapshot (guarded)
    try:
        import os
        debug_env = os.getenv("HIVEST_DEBUG_STOCK", "0").strip()
        if debug_env == "1" or getattr(options, "verbose", False):
            cm_dict = {
                k: v for k, v in cm.__dict__.items()
                if k in (
                    "cum_return","volatility","beta","sharpe","sortino","max_drawdown",
                    "rsi","sma20","sma50","sma200","pct_from_52w_high","pct_from_52w_low","last_price"
                ) and v is not None
            }
            bench_keys = list((benchmarks or {}).keys())
            fund_non_empty = [k for k, v in (fund or {}).items() if v not in (None, "", [], {}, 0)]
            print(f"[stock-debug] symbol={symbol} metrics_keys={list(cm_dict.keys())} bench={bench_keys} fund_keys_non_empty={fund_non_empty[:10]}")
    except Exception:
        pass

    return ctx
