from __future__ import annotations
import json
import argparse  # 1. Import argparse
from typing import List, Dict, Any

from .portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from .portfolio_analysis.processing.models import Holding, PortfolioInput, AnalysisOptions
from .portfolio_analysis.llm.prompts import build_portfolio_prompt
from .shared.llm_client import make_llm


# run_example() function can remain as it is for testing purposes
def run_example() -> str:
    holdings = [
        Holding(symbol="AAPL", weight=0.5, sector="Technology", avg_buy_price=175.0, bought_at="2024-06-01"),
        Holding(symbol="MSFT", weight=0.3, sector="Technology", avg_buy_price=320.0, bought_at="2024-07-15"),
        Holding(symbol="XOM", weight=0.2, sector="Energy", avg_buy_price=105.0, bought_at="2024-03-20"),
    ]
    portfolio_returns = [0.012, -0.006, 0.008]
    benchmark_returns = {"SPY": [0.010, -0.004, 0.006], "QQQ": [0.013, -0.007, 0.010]}
    per_symbol_returns = {"AAPL": [0.015, -0.010, 0.012], "MSFT": [0.011, -0.003, 0.006], "XOM": [0.005, 0.002, 0.004]}
    upcoming_events = [{"symbol": "AAPL", "type": "earnings", "date": "2025-10-25", "note": "Q4 call"},
                       {"symbol": "XOM", "type": "dividend", "date": "2025-09-20", "note": "ex-date"}]
    pi = PortfolioInput(holdings=holdings, timeframe_label="all time", portfolio_returns=portfolio_returns,
                        benchmark_returns=benchmark_returns, market_returns=benchmark_returns.get("SPY"),
                        per_symbol_returns=per_symbol_returns, risk_free_rate=0.0001, upcoming_events=upcoming_events)
    options = AnalysisOptions(top_k_winners_losers=5, concentration_threshold=0.15)
    analysis_result = analyze_portfolio(pi, options)
    instruction = (
        "You are an insightful financial analyst generating a JSON analysis. Your analysis **must be strictly based on the data provided** in the `calculatedMetrics` and `portfolioHoldings` objects.\n\n"
        "**Adhere to these rules:**\n"
        "1.  **For `portfolioAllocation`**: Create a narrative for each holding describing its role and sector.\n"
        "2.  **For `keyAdvantages`**: Identify strengths like sector balance, blend of growth/defense, and company quality.\n"
        "3.  **For `risksToWatch`**: Address concentration (using `hhiScore`), overweight sectors (using `sectorWeights`), and note the `missingSectors`.\n"
        "4.  **For `conclusionsAndRecommendations`**: Provide actionable advice based on the advantages and risks.\n"
        "5.  **For `bottomLine`**: Give a concise, final verdict on the portfolio's quality.\n\n"
        "Generate a JSON object with the following keys: `portfolioAllocation`, `keyAdvantages`, `risksToWatch`, `conclusionsAndRecommendations`, `bottomLine`."
    )
    llm = make_llm()
    prompt = instruction + "\n\n--- DATA ---\n" + json.dumps(analysis_result, separators=(",", ":"))
    return llm(prompt)


def run_from_user_input(positions, model: str | None = None, host: str | None = None):
    """
    This function now takes the user's positions and always uses a 'ytd' timeframe.
    """
    # 2. Force the timeframe to always be "year to date"
    timeframe = "ytd"

    holdings: List[Holding] = holdings_from_user_positions(positions)

    # auto_fill_series will populate the return data from FMP (via shared.market)
    pi = PortfolioInput(
        holdings=holdings,
        timeframe_label=timeframe,
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

    # This replaces the old analyze_portfolio call and the hardcoded instruction string
    analysis_context = analyze_portfolio(pi, options)
    
    # Use the sophisticated prompt builder instead of a simple instruction string
    prompt = build_portfolio_prompt(
        analysis_context["portfolio_input"],
        analysis_context["computed_metrics"],
        analysis_context["news_items"]
    )

    print("\n--- Sending Prompt to LLM ---")
    if options.verbose:
        print(prompt)
    print("---------------------------\n")

    llm = make_llm(options.llm_model, options.llm_host)
    return llm(prompt)


if __name__ == "__main__":
    # 3. Add argument parsing to read from the command line
    parser = argparse.ArgumentParser(description="Hivest Portfolio Analysis")
    parser.add_argument("--portfolio", type=str, required=True,
                        help="Portfolio holdings in the format 'TICKER1:WEIGHT1 TICKER2:WEIGHT2 ...'")
    args = parser.parse_args()

    positions = []
    # Parse the portfolio string into a list of dictionaries
    for holding in args.portfolio.split():
        try:
            symbol, weight_str = holding.split(':')
            weight = float(weight_str)
            positions.append({"symbol": symbol, "weight": weight})
        except ValueError:
            print(f"Warning: Skipping invalid holding format: {holding}")
            continue

    # 4. Call run_from_user_input instead of run_example
    if positions:
        # The 'timeframe' argument is no longer needed here
        analysis_json = run_from_user_input(positions)
        print("Here is the JSON analysis:\n")
        print(analysis_json)
    else:
        print("No valid portfolio holdings were provided.")