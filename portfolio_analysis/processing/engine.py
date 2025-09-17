from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import traceback

from .models import Holding, PortfolioInput, AnalysisOptions, ComputedMetrics
from ..llm.prompts import build_portfolio_prompt, _fallback_render, score_portfolio
from ...shared.market import auto_fill_series, infer_yahoo_range
from ...shared.performance import cumulative_return, benchmark_comparison, attribution_by_position, attribution_by_sector
from ...shared.risk import compute_volatility, compute_beta, compute_sharpe, compute_concentration, compute_drawdown_stats, compute_sortino
from ...shared.news import get_news_for_holdings
from ...shared.llm_client import make_llm


def holdings_from_user_positions(positions: List[Dict[str, Any]]) -> List[Holding]:
    """
    Accepts items like:
      {"symbol": "AAPL", "weight_pct": 25.0, "avg_buy_price": 180.0, "bought_at": "2024-10-15"}
    Or:
      {"symbol": "AAPL", "weight": 0.25, "avg_buy_price": 180.0}
    Returns a list[Holding] with normalized fractional weights.
    """
    hs: List[Holding] = []
    raw_weights: List[float] = []
    for p in positions or []:
        if "weight" in p and p["weight"] is not None:
            raw_weights.append(float(p["weight"]))
        else:
            raw_weights.append(float(p.get("weight_pct", 0.0)) / 100.0)
    total = sum(w for w in raw_weights if w > 0) or 1.0
    norm = [max(0.0, w) / total for w in raw_weights]

    for p, w in zip(positions or [], norm):
        hs.append(Holding(
            symbol=(p.get("symbol") or "").upper(),
            weight=w,
            avg_buy_price=p.get("avg_buy_price"),
            bought_at=p.get("bought_at"),
            sector=p.get("sector"),
            name=p.get("name"),
        ))
    return hs


def analyze_portfolio(pi: PortfolioInput, options: Optional[AnalysisOptions] = None) -> str:
    options = options or AnalysisOptions()

    # Local logger toggle
    LOG = print if getattr(options, "verbose", False) else (lambda *a, **k: None)

    # Inputs echo
    if getattr(options, "echo_tools", True):
        try:
            LOG("[inputs] symbols=", [h.symbol for h in (pi.holdings or [])])
            LOG(f"[inputs] timeframe='{pi.timeframe_label}'")
            LOG("[inputs] weights(fractions)=", {h.symbol: float(h.weight or 0.0) for h in (pi.holdings or [])})
            if any(getattr(h, "avg_buy_price", None) is not None for h in (pi.holdings or [])):
                LOG("[inputs] avg_buy_price provided for:", [h.symbol for h in pi.holdings if getattr(h, "avg_buy_price", None) is not None])
        except Exception:
            pass

    # Auto-fill missing series from market data if needed
    try:
        needs_fill = not (pi.portfolio_returns and pi.benchmark_returns and pi.per_symbol_returns)
        if needs_fill:
            try:
                yf_range = infer_yahoo_range(getattr(pi, "timeframe_label", "6m"))
            except Exception:
                yf_range = "auto"
            LOG(f"[market] using Yahoo Finance public chart API (range={yf_range}) to fill series…")
            auto_fill_series(pi)
            LOG(f"[market] per-symbol returns filled for "
                f"{sum(1 for v in (pi.per_symbol_returns or {}).values() if v)} symbols; "
                f"portfolio series length={len(pi.portfolio_returns or [])}")
    except Exception:
        pass

    # Collect symbols, sectors, and weights
    symbols = [h.symbol.upper() for h in (pi.holdings or [])]
    weights = {h.symbol.upper(): float(h.weight or 0.0) for h in (pi.holdings or [])}

    # --- performance & attribution ---
    LOG("[performance] computing cumulative return and benchmark comparison…")
    cum_ret = cumulative_return(pi.portfolio_returns)
    bench = benchmark_comparison(pi.portfolio_returns, pi.benchmark_returns or {})

    LOG("[attribution] by-position and by-sector attribution…")
    pos_attr = attribution_by_position([vars(h) for h in pi.holdings], pi.per_symbol_returns)
    sec_attr = attribution_by_sector([vars(h) for h in pi.holdings], pi.per_symbol_returns)

    # --- risk metrics ---
    LOG("[risk] computing volatility, beta, Sharpe, Sortino, and drawdown stats…")
    vol = compute_volatility(pi.portfolio_returns)
    market = pi.market_returns or (pi.benchmark_returns.get("SPY") if pi.benchmark_returns else None) or []
    beta = compute_beta(pi.portfolio_returns, market)
    sharpe = compute_sharpe(pi.portfolio_returns, risk_free_rate=pi.risk_free_rate)

    dd_stats = compute_drawdown_stats(pi.portfolio_returns)
    sortino = compute_sortino(pi.portfolio_returns)

    ppy = float(getattr(pi, "periods_per_year", 252.0)) or 252.0
    vol_annual = vol * (ppy ** 0.5)
    sharpe_annual = sharpe * (ppy ** 0.5)
    sortino_annual = sortino * (ppy ** 0.5)

    conc = compute_concentration(weights, threshold=options.concentration_threshold)

    cm = ComputedMetrics(
        cum_return=cum_ret,
        benchmarks=bench["benchmarks"],
        by_symbol=pos_attr["by_symbol"],
        winners=pos_attr["winners"],
        losers=pos_attr["losers"],
        by_sector=sec_attr["by_sector"],
        top_sectors=sec_attr["top"],
        bottom_sectors=sec_attr["bottom"],
        volatility=vol,
        beta=beta,
        sharpe=sharpe,
        concentration=conc,
        max_drawdown=dd_stats["max_drawdown"],
        calmar_like=dd_stats["calmar_like"],
        sortino=sortino,
        volatility_annual=vol_annual,
        sharpe_annual=sharpe_annual,
        sortino_annual=sortino_annual,
    )

    # Resources: news / briefs
    news_items: list[dict] = []
    if getattr(options, "include_news", True):
        LOG("[news] fetching recent news (GNews/MediaStack) with fallback to Ollama briefs.")
        try:
            news_items = get_news_for_holdings([vars(h) for h in pi.holdings], limit=options.news_limit) or []
        except Exception:
            news_items = []

    setattr(pi, 'include_events', bool(options.include_events))
    prompt = build_portfolio_prompt(pi, cm, news_items=news_items)

    # LLM call with graceful fallback
    llm = None
    try:
        llm = make_llm(getattr(options, "llm_model", None), getattr(options, "llm_host", None))
    except Exception:
        llm = None

    if callable(llm):
        LOG("[llm] rendering summary via chat completion.")
        try:
            text = (llm(prompt) or "").strip()
            # Ensure score hint correctness by recomputing (uses cm values)
            try:
                score_info = score_portfolio(cm)
            except Exception:
                score_info = {"score": 7, "reasons": "balanced risk/return"}
            # We no longer force-insert score line; keep model output as-is
            if text:
                return text
        except Exception:
            print(f"\n[LLM_ERROR] The LLM call failed. The specific error is:\n---")
            traceback.print_exc()
            print("---\n")
            pass
    else:
        LOG("[fallback] rendering deterministic summary (no LLM).")

    # Deterministic paragraph fallback
    return _fallback_render(pi, cm, news_items=news_items)
