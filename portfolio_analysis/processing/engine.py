"""hivest/portfolio_analysis/processing/engine.py"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import traceback
import json
import yfinance as yf

from .models import Holding, PortfolioInput, AnalysisOptions, ComputedMetrics
from ...shared.market import auto_fill_series
from ...shared.performance import cumulative_return, benchmark_comparison, attribution_by_position, attribution_by_sector
from ...shared.risk import compute_volatility, compute_beta, compute_sharpe, compute_concentration, compute_drawdown_stats, compute_sortino
from ...shared.news import get_news_for_holdings


def holdings_from_user_positions(positions: List[Dict[str, Any]]) -> List[Holding]:
    """
    Accepts items like:
      {"symbol": "AAPL", "weight_pct": 25.0, "avg_buy_price": 180.0, "bought_at": "2024-10-15"}
    Or:
      {"symbol": "AAPL", "weight": 0.25, "avg_buy_price": 180.0}
    Returns a list[Holding] with normalized fractional weights, enriched with sector and name.
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
        symbol = (p.get("symbol") or "").upper()
        holding = Holding(
            symbol=symbol,
            weight=w,
            avg_buy_price=p.get("avg_buy_price"),
            bought_at=p.get("bought_at"),
        )

        # --- ADD THIS BLOCK ---
        # Fetch company name and sector from yfinance
        try:
            ticker_info = yf.Ticker(symbol).info
            holding.name = ticker_info.get('longName', symbol)
            holding.sector = ticker_info.get('sector', 'Unknown')
            print(f"Successfully fetched info for {symbol}: Sector='{holding.sector}', Name='{holding.name}'")
        except Exception as e:
            print(f"Could not fetch info for {symbol}: {e}. Setting to defaults.")
            holding.name = symbol
            holding.sector = "Unknown"
        # --- END BLOCK ---

        hs.append(holding)

    return hs




def analyze_portfolio(pi: PortfolioInput, options: Optional[AnalysisOptions] = None) -> Dict[str, Any]:
    options = options or AnalysisOptions()
    holdings_as_dicts = [{"symbol": h.symbol, "weight": h.weight, "sector": h.sector} for h in pi.holdings]

    # 1. Auto-fill all time-series data (returns, benchmarks) using Yahoo Finance
    if options.verbose:
        print("Fetching market data for portfolio and benchmarks...")
    auto_fill_series(pi)

    # 2. Calculate core performance and risk metrics
    if options.verbose:
        print("Calculating performance and risk metrics...")
    cum_ret = cumulative_return(pi.portfolio_returns)
    vol = compute_volatility(pi.portfolio_returns)
    beta = compute_beta(pi.portfolio_returns, pi.market_returns)
    sharpe = compute_sharpe(pi.portfolio_returns, pi.risk_free_rate)
    sortino = compute_sortino(pi.portfolio_returns)
    drawdown_stats = compute_drawdown_stats(pi.portfolio_returns)
    
    # 3. Perform attribution analysis
    pos_attr = attribution_by_position(holdings_as_dicts, pi.per_symbol_returns)
    sec_attr = attribution_by_sector(holdings_as_dicts, pi.per_symbol_returns)

    # 4. Analyze concentration
    weights_map = {h.symbol: h.weight for h in pi.holdings}
    concentration = compute_concentration(weights_map, options.concentration_threshold)

    # 5. Get news if requested
    news_items = []
    if options.include_news:
        if options.verbose:
            print(f"Fetching up to {options.news_limit} news articles...")
        news_items = get_news_for_holdings(holdings_as_dicts, limit=options.news_limit)

    # 6. Assemble the final ComputedMetrics object to be used by the prompt builder
    # Note: The 'ComputedMetrics' model already exists in 'models.py'
    computed_metrics = ComputedMetrics(
        cum_return=cum_ret,
        benchmarks=benchmark_comparison(pi.portfolio_returns, pi.benchmark_returns).get("benchmarks", {}),
        by_symbol=pos_attr["by_symbol"],
        winners=pos_attr["winners"],
        losers=pos_attr["losers"],
        by_sector=sec_attr["by_sector"],
        top_sectors=sec_attr["top"],
        bottom_sectors=sec_attr["bottom"],
        volatility=vol,
        beta=beta,
        sharpe=sharpe,
        sortino=sortino,
        concentration=concentration,
        max_drawdown=drawdown_stats["max_drawdown"],
        calmar_like=drawdown_stats["calmar_like"]
    )
    
    # This dictionary will be the context for the LLM prompt
    return {
        "portfolio_input": pi,
        "computed_metrics": computed_metrics,
        "news_items": news_items
    }
