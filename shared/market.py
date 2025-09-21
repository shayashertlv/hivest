"""hivest/shared/market.py"""
from __future__ import annotations

"""Market data fetching and return-series utilities for Hivest.
- Uses Financial Modeling Prep (FMP) for historical prices via requests.
- Builds aligned daily close series and simple returns for symbols and benchmarks.

Key functions:
- infer_yahoo_range(timeframe_label: str) -> str
- fetch_yahoo_chart(symbol: str, yf_range: str = '6mo', interval: str = '1d') -> (dates, closes)
- compute_simple_returns(closes: List[float]) -> List[float]
- align_by_dates(series_map: Dict[str, Tuple[List[str], List[float]]]) -> Dict[str, List[float]]
- build_per_symbol_returns(symbols: List[str], timeframe_label: str) -> Dict[str, List[float]]
- auto_fill_series(pi: 'PortfolioInput') -> 'PortfolioInput' (mutates and returns pi)

Notes:
- Symbols are used as provided. Common ETFs like SPY/QQQ/QQQM typically work.
- We compute portfolio returns as a constant-weight daily blend: sum_i(w_i * r_i,t).
- Benchmarks: at least SPY is filled; QQQ is added if available.
"""
import os, datetime as _dt, re
import requests
from typing import Dict, List, Tuple, Optional

_session = requests.Session()


import re
from typing import Optional

def infer_yahoo_range(label: Optional[str]) -> str:
    s = (label or '').strip().lower()
    if not s or s in ('all time', 'alltime', 'lifetime'):
        return '1y'
    # last N months pattern
    m = re.match(r"^(last\s+)?(\d{1,2})\s*months?", s)
    if m:
        n = int(m.group(2))
        if n <= 1:
            return '1mo'
        if n <= 3:
            return '3mo'
        if n <= 6:
            return '6mo'
        if n <= 12:
            return '1y'
        if n <= 24:
            return '2y'
        return '5y'
    # common aliases
    if s in ('ytd', 'year to date', 'year-to-date'):
        return 'ytd'
    if s in ('1m', '1 month'):
        return '1mo'
    if s in ('3m', '3 months'):
        return '3mo'
    if s in ('4m', 'four months', 'last four months', 'last 4 months'):
        return '6mo'
    if s in ('6m', '6 months'):
        return '6mo'
    if s in ('1y', '12m', '12 months'):
        return '1y'
    if s in ('2y',):
        return '2y'
    # friendly aliases
    if s in ('6 months', 'last six months'):
        return '6mo'
    if s in ('12 months', 'last 12 months', '1 year'):
        return '1y'
    return '6mo'


def fetch_yahoo_chart(symbol: str, yf_range: str = '6mo', interval: str = '1d') -> Tuple[List[str], List[float]]:
    """
    Provider-neutral price fetcher implemented via Financial Modeling Prep (FMP).
    Keeps the legacy name/signature for compatibility.
    """
    # Map common range labels to a start date for FMP
    try:
        today = _dt.date.today()
        s = (yf_range or '6mo').lower()
        if s == 'ytd':
            start = _dt.date(today.year, 1, 1)
        else:
            months = {
                '1mo': 1, '3mo': 3, '6mo': 6,
                '1y': 12, '2y': 24, '5y': 60,
            }.get(s, 6)
            # approx 30.44 days/month
            start = today - _dt.timedelta(days=int(months * 30.44))
    except Exception:
        start = _dt.date.today() - _dt.timedelta(days=180)
        today = _dt.date.today()

    fmp = (os.getenv('FMP_FREE_API_KEY') or '').strip()
    if not fmp:
        return ([], [])

    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol.upper()}"
    params = {"from": start.isoformat(), "to": today.isoformat(), "apikey": fmp, "serietype": "line"}
    try:
        r = _session.get(url, params=params, timeout=10)
        r.raise_for_status()
        js = r.json() or {}
        hist = js.get('historical') or []
    except Exception as e:
        print(f"[market] FMP fetch failed for {symbol}: {e}")
        return ([], [])

    # FMP returns most-recent-first; sort ascending by date
    try:
        rows = sorted((x for x in hist if x.get('close') is not None and x.get('date')), key=lambda x: x['date'])
    except Exception:
        rows = []
    dates = [str(x['date']) for x in rows]
    prices = [float(x['close']) for x in rows]
    return dates, prices


def compute_simple_returns(closes: List[float]) -> List[float]:
    rets: List[float] = []
    for i in range(1, len(closes)):
        p0 = closes[i-1]
        p1 = closes[i]
        if p0 and p1:
            try:
                rets.append(p1 / p0 - 1.0)
            except Exception:
                rets.append(0.0)
        else:
            rets.append(0.0)
    return rets


def align_by_dates(series_map: Dict[str, Tuple[List[str], List[float]]]) -> Dict[str, List[float]]:
    """
    Align per-symbol close series by an anchor symbol's calendar (prefer 'SPY' if present),
    then return per-symbol simple returns (all the same length). Symbols that cannot align
    to the anchor's dates are dropped.

    Input:  symbol -> (dates[YYYY-MM-DD...], closes[float...])
    Output: symbol -> simple_returns[float...]   # aligned across symbols
    """
    # Keep only usable candidates (>=2 prices and matching date/price lengths)
    candidates = {sym: (ds, ps) for sym, (ds, ps) in series_map.items()
                  if isinstance(ds, list) and isinstance(ps, list)
                  and len(ps) >= 2 and len(ds) == len(ps)}
    if not candidates:
        return {}

    # Choose anchor: prefer SPY, else the symbol with the longest price history
    if "SPY" in candidates:
        anchor_sym = "SPY"
    else:
        anchor_sym = max(candidates.keys(), key=lambda k: len(candidates[k][0]))

    ds_anchor, ps_anchor = candidates[anchor_sym]

    # Ensure anchor dates are strictly increasing and paired with prices
    # (Provider data is usually sorted, but we guard anyway)
    anc_pairs = sorted(zip(ds_anchor, ps_anchor), key=lambda x: x[0])
    ds_anchor_sorted = [d for d, _ in anc_pairs]

    out: Dict[str, List[float]] = {}

    # Build aligned price vectors on the anchor's date grid
    for sym, (ds, ps) in candidates.items():
        # Map this symbol's dates to price
        idx = {d: p for d, p in zip(ds, ps)}
        aligned_prices: List[float] = []
        started = False
        last_price: Optional[float] = None
        missing = 0
        # Build aligned series with limited forward-fill after first common date
        for d in ds_anchor_sorted:
            p = idx.get(d)
            if p is None:
                if not started:
                    # We haven't reached the first overlapping date yet; skip until first match
                    continue
                if last_price is None:
                    # No prior price to carry forward — cannot align further
                    break
                # Forward-fill missing date
                missing += 1
                aligned_prices.append(last_price)
            else:
                started = True
                last_price = float(p)
                aligned_prices.append(last_price)
        # Accept if we have enough data and missing ratio is <= 5%
        total_points = len(aligned_prices)
        if total_points >= 2:
            missing_ratio = (missing / total_points) if total_points > 0 else 1.0
            if missing_ratio <= 0.05:
                out[sym] = compute_simple_returns(aligned_prices)

    if not out:
        return {}

    # Clip all to the shortest length (defensive)
    min_len = min(len(v) for v in out.values())
    if min_len <= 0:
        return {}
    for sym in list(out.keys()):
        out[sym] = out[sym][-min_len:]

    return out



def build_per_symbol_returns(symbols: List[str], timeframe_label: str) -> Dict[str, List[float]]:
    yf_range = infer_yahoo_range(timeframe_label)
    series: Dict[str, Tuple[List[str], List[float]]] = {}
    for s in symbols or []:
        try:
            ds, ps = fetch_yahoo_chart(s, yf_range=yf_range, interval='1d')
            if ds and ps:
                series[s.upper()] = (ds, ps)
        except Exception:
            # skip symbol on failure
            continue
    return align_by_dates(series)



def auto_fill_series(pi) -> object:
    """
    Populate missing series in a PortfolioInput in-place using market data (FMP).
    - per_symbol_returns: built for the portfolio symbols
    - portfolio_returns: weighted blend of per-symbol daily returns
    - benchmark_returns: ensure at least SPY (and QQQ if available) is present
    - market_returns: set to SPY
    Returns the same pi for chaining.
    """
    try:
        # Collect symbols and weights
        holdings = pi.holdings or []
        symbols = [h.symbol.upper() for h in holdings]
        weights = {h.symbol.upper(): float(h.weight or 0.0) for h in holdings}
        # Per-symbol returns
        sym_rets = build_per_symbol_returns(symbols, getattr(pi, 'timeframe_label', '6m'))
        if sym_rets:
            pi.per_symbol_returns = sym_rets
            # Align weights to available symbols
            avail_syms = list(sym_rets.keys())
            # Normalize weights of available symbols to sum to 1 for the blend
            w_vals = [max(0.0, weights.get(s, 0.0)) for s in avail_syms]
            w_sum = sum(w_vals) or 1.0
            w_norm = [w / w_sum for w in w_vals]
            # Blend portfolio daily returns
            n = len(next(iter(sym_rets.values())))
            port: List[float] = []
            for t in range(n):
                port.append(sum(w_norm[i] * sym_rets[avail_syms[i]][t] for i in range(len(avail_syms))))
            pi.portfolio_returns = port
        # Benchmarks: SPY and QQQ
        br = dict(getattr(pi, 'benchmark_returns', {}) or {})
        if 'SPY' not in br:
            br_syms = ['SPY']
        else:
            br_syms = []
        if 'QQQ' not in br:
            br_syms.append('QQQ')
        if br_syms:
            bench = build_per_symbol_returns(br_syms, getattr(pi, 'timeframe_label', '6m'))
            for k, v in bench.items():
                br[k] = v
        pi.benchmark_returns = br
        # Market proxy = SPY
        pi.market_returns = br.get('SPY') or []

    except Exception as e:
        # On any error, print a warning and leave pi as-is.
        print(f"\n[ERROR] Failed to fetch or align market data: {e}\n")
        pass
    return pi
