from __future__ import annotations

"""hivest/main.py"""
"""
Hivest main orchestrator.
Provides a simple example of how to assemble inputs and run the portfolio analysis.
"""

from typing import List, Dict, Any

from .portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from .portfolio_analysis.processing.models import Holding, PortfolioInput, AnalysisOptions


def run_example() -> str:
    # Example dummy data
    holdings = [
        Holding(symbol="AAPL", weight=0.5, sector="Technology", avg_buy_price=175.0, bought_at="2024-06-01"),
        Holding(symbol="MSFT", weight=0.3, sector="Technology", avg_buy_price=320.0, bought_at="2024-07-15"),
        Holding(symbol="XOM", weight=0.2, sector="Energy", avg_buy_price=105.0, bought_at="2024-03-20"),
    ]

    # Three periods of returns as an illustration
    portfolio_returns = [0.012, -0.006, 0.008]
    benchmark_returns = {
        "SPY": [0.010, -0.004, 0.006],
        "QQQ": [0.013, -0.007, 0.010],
    }

    per_symbol_returns = {
        "AAPL": [0.015, -0.010, 0.012],
        "MSFT": [0.011, -0.003, 0.006],
        "XOM": [0.005, 0.002, 0.004],
    }

    upcoming_events = [
        {"symbol": "AAPL", "type": "earnings", "date": "2025-10-25", "note": "Q4 call"},
        {"symbol": "XOM", "type": "dividend", "date": "2025-09-20", "note": "ex-date"},
    ]

    pi = PortfolioInput(
        holdings=holdings,
        timeframe_label="YTD",
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        market_returns=benchmark_returns.get("SPY"),
        per_symbol_returns=per_symbol_returns,
        risk_free_rate=0.0001,
        upcoming_events=upcoming_events,
    )

    options = AnalysisOptions(top_k_winners_losers=5, concentration_threshold=0.15)

    return analyze_portfolio(pi, options)


def run_from_user_input(positions, timeframe="YTD", model: str | None = None, host: str | None = None):
    """
    positions: list of dicts describing the portfolio. Supported keys per entry:
        {symbol, cut?, weight?, weight_pct?, avg_buy_price?, bought_at?, sector?, name?}
        "cut" represents the portfolio slice (0..1). Weight/weight_pct remain supported for
        backward compatibility.
    timeframe: ignored — the analysis is always executed on a Year-to-Date window.
    """
    holdings: List[Holding] = holdings_from_user_positions(positions)
    pi = PortfolioInput(
        holdings=holdings,
        timeframe_label="YTD",
        # Leave series empty; auto_fill_series will populate from Yahoo
        portfolio_returns=[],
        benchmark_returns={},
        market_returns=[],
        per_symbol_returns={},
        risk_free_rate=0.0001,
        upcoming_events=[],
    )

    options = AnalysisOptions(
        top_k_winners_losers=5,
        concentration_threshold=0.15,
        include_news=True,
        news_limit=6,
        include_events=True,
        verbose=True,
        echo_tools=True,
        llm_model=model,
        llm_host=host,
    )

    return analyze_portfolio(pi, options)


if __name__ == "__main__":
    print(run_example())
