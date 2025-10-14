from __future__ import annotations
from typing import Optional
from .models import StockInput, StockOptions, StockMetrics, StockReport, EtfProfile
from .indicators import compute_rsi, sma, pct_from_window_extrema
from ...shared.market import fetch_yahoo_chart, compute_simple_returns, infer_yahoo_range, align_by_dates
from ...shared.performance import cumulative_return
from ...shared.risk import compute_beta, compute_volatility, compute_drawdown_stats
from ...shared.fundamentals import get_fundamentals
from ...shared.news import fetch_news_api
from ...shared.events import next_earnings_date
from ..llm.prompts import build_stock_prompt
from ...shared.llm_client import make_llm
import json

def _load_series(symbol: str, tf_label: str):
    yf_range = infer_yahoo_range(tf_label)
    d, c = fetch_yahoo_chart(symbol, yf_range=yf_range, interval="1d")
    return d, c, compute_simple_returns(c)


def analyze_stock(si: StockInput, opt: Optional[StockOptions] = None) -> StockReport:
    opt = opt or StockOptions()
    LOG = print if opt.verbose else (lambda *a, **k: None)

    LOG(f"[inputs] symbol={si.symbol} timeframe='{si.timeframe_label}'")
    dates, closes, rets = _load_series(si.symbol, si.timeframe_label)

    spy_dates, spy_closes = fetch_yahoo_chart("SPY", yf_range=infer_yahoo_range(si.timeframe_label), interval="1d")
    aligned = align_by_dates({"SPY": (spy_dates, spy_closes), si.symbol.upper(): (dates, closes)})
    srets = aligned.get(si.symbol.upper(), rets)
    spyrets = compute_simple_returns(spy_closes)

    cum_ret = cumulative_return(srets)
    vol = compute_volatility(srets)
    beta = compute_beta(srets, spyrets)
    dd = compute_drawdown_stats(srets)["max_drawdown"]
    rsi14 = compute_rsi(closes, 14)
    s20, s50, s200 = sma(closes, 20), sma(closes, 50), sma(closes, 200)
    pct_hi, pct_lo = pct_from_window_extrema(closes, 252)

    window_52w = closes[-252:] if len(closes) >= 252 else closes
    high_52w = max(window_52w) if window_52w else (closes[-1] if closes else 0.0)
    low_52w = min(window_52w) if window_52w else (closes[-1] if closes else 0.0)

    # --- Start of Corrected Section ---
    # Fetch all data first
    inst_type, fundamentals, etf_profile_data = get_fundamentals(si.symbol)
    social_sentiment_data = fundamentals.pop("social_sentiment", None)

    news_items = fetch_news_api([si.symbol], limit=opt.news_limit) if opt.include_news else []
    market_news_items = fetch_news_api(['SPY'], limit=3) if opt.include_news else []
    nxt = next_earnings_date(si.symbol) if opt.include_events else None

    etf_profile = EtfProfile(**etf_profile_data) if inst_type == "etf" and isinstance(etf_profile_data, dict) else None

    # Now, create the metrics object with all the defined variables
    metrics = StockMetrics(
        dates=dates, closes=closes, returns=srets, cum_return=cum_ret, volatility=vol,
        beta_vs_spy=beta, max_drawdown=dd, rsi14=rsi14, sma20=s20, sma50=s50, sma200=s200,
        pct_from_52w_high=pct_hi, pct_from_52w_low=pct_lo,
        high_52w=high_52w,
        low_52w=low_52w,
        last_close=closes[-1] if closes else 0.0,
        fundamentals=fundamentals,
        social_sentiment=social_sentiment_data,
        news_items=news_items,
        market_news_items=market_news_items,
        next_earnings=nxt,
        instrument_type=inst_type,
        etf_profile=etf_profile
    )
    # --- End of Corrected Section ---

    prompt = build_stock_prompt(si.symbol, metrics)
    llm = make_llm(opt.llm_model, opt.llm_host)

    # This line was also referencing an old function, correcting it.
    raw_json = (llm(prompt) or "").strip()

    # A simple fallback for the narrative part
    try:
        narrative = json.loads(raw_json)
    except json.JSONDecodeError:
        narrative = {"error": "Failed to generate valid analysis."}

    return StockReport(metrics=metrics, narrative=narrative)


def _fallback(m: StockMetrics) -> str:
    parts = []
    parts.append(f"Return {m.cum_return*100:.2f}% with vol {m.volatility*100:.2f}% and beta {m.beta_vs_spy:.2f}.")
    parts.append(f"RSI {m.rsi14:.1f}; price {m.pct_from_52w_high*100:.2f}% off 52w high.")
    if m.fundamentals:
        pe = m.fundamentals.get("pe_ttm"); mc = m.fundamentals.get("market_cap")
        f = []
        if pe is not None: f.append(f"P/E≈{pe:.1f}")
        if mc is not None: f.append(f"mkt cap≈{mc/1e9:.1f}B")
        if f: parts.append("Valuation: " + ", ".join(f) + ".")
    if m.next_earnings: parts.append(f"Next earnings on {m.next_earnings}.")
    return " ".join(parts)
