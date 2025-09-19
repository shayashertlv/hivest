from __future__ import annotations

"""hivest/main.py"""
"""
Hivest main orchestrator.
Provides a simple example of how to assemble inputs and run the portfolio analysis.
"""

from typing import List, Dict, Any
import json

from .portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from .portfolio_analysis.processing.models import Holding, PortfolioInput, AnalysisOptions
from .shared.llm_client import make_llm


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
        timeframe_label="all time",
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        market_returns=benchmark_returns.get("SPY"),
        per_symbol_returns=per_symbol_returns,
        risk_free_rate=0.0001,
        upcoming_events=upcoming_events,
    )

    options = AnalysisOptions(top_k_winners_losers=5, concentration_threshold=0.15)

    analysis_result = analyze_portfolio(pi, options)

    instruction = (
      "You are an insightful financial analyst generating a JSON analysis. Your analysis **must be strictly based on the data provided** in the `calculatedMetrics` and `portfolioHoldings` objects.\n\n"
      "**Adhere to these rules:**\n"
      "1.  **For `keyAdvantages`**: Focus only on the quality of the companies listed in `portfolioHoldings`.\n"
      "2.  **For `risksToWatch`**: Your first point **must** address concentration. Use the `hhiScore` to determine the severity (a score over 2500 means 'extreme concentration'). Your second point **must** identify the overweight sector using the `sectorWeights` data. Your third point **must** mention some of the `missingSectors`.\n"
      "3.  **For `bottomLine`**: You are **forbidden** from calling the portfolio 'well-balanced' or 'diversified' if the `hhiScore` is above 1500. Instead, describe it as 'concentrated' or 'highly focused'.\n\n"
      "Generate a JSON object with the following keys: `portfolioAllocation`, `keyAdvantages`, `risksToWatch`, `conclusionsAndRecommendations`, `bottomLine`."
    )

    llm = make_llm()
    prompt = instruction + "\n\n--- DATA ---\n" + json.dumps(analysis_result, separators=(",", ":"))
    return llm(prompt)


def run_from_user_input(positions, timeframe="6m", model: str | None = None, host: str | None = None):
    """
    positions: list of dicts with either {symbol, weight_pct, avg_buy_price?, bought_at?}
               or {symbol, weight (0..1), avg_buy_price?, bought_at?}
    timeframe: e.g. '1m', '3m', '6m', '1y', 'ytd', or 'last 6 months'
    """
    holdings: List[Holding] = holdings_from_user_positions(positions)
    pi = PortfolioInput(
        holdings=holdings,
        timeframe_label=timeframe,
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

    analysis_result = analyze_portfolio(pi, options)

    instruction = (
      "You are an insightful financial analyst generating a JSON analysis. Your analysis **must be strictly based on the data provided** in the `calculatedMetrics` and `portfolioHoldings` objects.\n\n"
      "**Adhere to these rules:**\n"
      "1.  **For `keyAdvantages`**: Focus only on the quality of the companies listed in `portfolioHoldings`.\n"
      "2.  **For `risksToWatch`**: Your first point **must** address concentration. Use the `hhiScore` to determine the severity (a score over 2500 means 'extreme concentration'). Your second point **must** identify the overweight sector using the `sectorWeights` data. Your third point **must** mention some of the `missingSectors`.\n"
      "3.  **For `bottomLine`**: You are **forbidden** from calling the portfolio 'well-balanced' or 'diversified' if the `hhiScore` is above 1500. Instead, describe it as 'concentrated' or 'highly focused'.\n\n"
      "Generate a JSON object with the following keys: `portfolioAllocation`, `keyAdvantages`, `risksToWatch`, `conclusionsAndRecommendations`, `bottomLine`."
    )

    llm = make_llm(model, host)
    prompt = instruction + "\n\n--- DATA ---\n" + json.dumps(analysis_result, separators=(",", ":"))
    return llm(prompt)


if __name__ == "__main__":
    print(run_example())
