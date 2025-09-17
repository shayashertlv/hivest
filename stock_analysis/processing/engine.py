from __future__ import annotations
from typing import List, Optional
from .models import StockInput, StockOptions, StockMetrics, StockReport
from .indicators import compute_rsi, sma, pct_from_window_extrema
from ...shared.market import fetch_yahoo_chart, compute_simple_returns, infer_yahoo_range, align_by_dates
from ...shared.performance import cumulative_return
from ...shared.risk import compute_beta, compute_volatility, compute_drawdown_stats
from ...shared.fundamentals import get_fundamentals
from ...shared.news import fetch_news_api
from ...shared.events import next_earnings_date
from ..llm.prompts import build_stock_prompt
from ...shared.llm_client import make_llm


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

    fundamentals = get_fundamentals(si.symbol)
    news_items = fetch_news_api([si.symbol], limit=opt.news_limit) if opt.include_news else []
    nxt = next_earnings_date(si.symbol) if opt.include_events else None

    metrics = StockMetrics(
        dates=dates, closes=closes, returns=srets, cum_return=cum_ret, volatility=vol,
        beta_vs_spy=beta, max_drawdown=dd, rsi14=rsi14, sma20=s20, sma50=s50, sma200=s200,
        pct_from_52w_high=pct_hi, pct_from_52w_low=pct_lo,
        fundamentals=fundamentals, news_items=news_items, next_earnings=nxt
    )

    prompt = build_stock_prompt(si.symbol, si.timeframe_label, metrics)
    llm = make_llm(opt.llm_model, opt.llm_host)
    narrative = (llm(prompt) or "").strip() if callable(llm) else _fallback(si.symbol, si.timeframe_label, metrics)
    return StockReport(metrics=metrics, narrative=narrative)


def _fallback(symbol: str, tf_label: str, m: StockMetrics) -> str:
    """Deterministic structured summary mirroring the LLM format."""

    snapshot_lines = []
    snapshot_lines.append(
        f"{symbol.upper()} → {tf_label} return {m.cum_return*100:.1f}% with beta {m.beta_vs_spy:.2f} and vol {m.volatility*100:.1f}%."
    )
    snapshot_lines.append(
        f"Momentum check: RSI {m.rsi14:.1f}; trading {m.pct_from_52w_high*100:.1f}% from 52w high / {m.pct_from_52w_low*100:.1f}% from low."
    )

    advantages = []
    if m.cum_return >= 0:
        advantages.append(f"Positive YTD performance {m.cum_return*100:.1f}% with controlled drawdown {m.max_drawdown*100:.1f}%.")
    else:
        advantages.append(f"Underperformance {m.cum_return*100:.1f}% creates potential mean-reversion if catalysts land.")
    if m.fundamentals.get("pe_ttm"):
        advantages.append(f"Valuation marker: P/E ≈{m.fundamentals['pe_ttm']:.1f} with market cap ≈{(m.fundamentals.get('market_cap', 0)/1e9):.1f}B.")
    if m.news_items:
        advantages.append(f"Recent focus: {m.news_items[0].get('title','headline')} ({m.news_items[0].get('source','')}).")

    risks = []
    if m.beta_vs_spy > 1.1:
        risks.append(f"High beta {m.beta_vs_spy:.2f} could amplify drawdowns.")
    if m.max_drawdown < -0.2:
        risks.append(f"History of deep pullbacks (max drawdown {m.max_drawdown*100:.1f}%).")
    risks.append(f"RSI {m.rsi14:.1f} signals {'overbought' if m.rsi14 > 70 else 'neutral' if 30 <= m.rsi14 <= 70 else 'oversold'} zone.")

    outlook = []
    outlook.append("Monitor AI/data demand, revenue growth cadence, and margin progression for direction.")
    if m.next_earnings:
        outlook.append(f"Next earnings checkpoint: {m.next_earnings} (watch guidance and bookings).")
    outlook.append("Track capex plans and supply chain updates for clues on capacity scaling.")

    conclusions = []
    conclusions.append("Base case: hold/add on weakness if execution stays on track.")
    conclusions.append(f"Valuation watch: {('premium multiple' if m.fundamentals.get('pe_ttm', 0) > 25 else 'reasonable multiple')} vs growth outlook.")

    bottom_line = (
        "💡 Bottom line: Focus on execution vs. capex ambitions and demand conversion over the next year to gauge upside vs. volat"
        "ility."
    )

    return (
        "📊 Company Snapshot\n"
        + "\n".join(snapshot_lines)
        + "\n\n⚖️ Key Advantages\n"
        + "\n".join(f"- {ln}" for ln in advantages)
        + "\n\n⚠️ Risks to Watch\n"
        + "\n".join(f"- {ln}" for ln in risks)
        + "\n\n🔮 12–18-Month Outlook (what could drive upside/downside)\n"
        + "\n".join(f"- {ln}" for ln in outlook)
        + "\n\n📝 Conclusions & Recommendations\n"
        + "\n".join(f"- {ln}" for ln in conclusions)
        + f"\n\n{bottom_line}"
    )
