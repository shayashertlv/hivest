"""hivest/portfolio_analysis/processing/engine.py"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import traceback
import json

import yfinance as yf

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


# Add this helper function inside engine.py

def _calculate_sector_weights(holdings: dict) -> dict:
    """
    Calculates the total weight of each sector in the portfolio.
    """
    sector_weights: Dict[str, float] = {}
    print("Fetching sector data for each holding...")
    for ticker_symbol, data in holdings.items():
        try:
            ticker = yf.Ticker(ticker_symbol)
            # Use .info which is more robust than .get_info()
            sector = ticker.info.get('sector', 'Unknown')

            if sector not in sector_weights:
                sector_weights[sector] = 0.0
            # Assuming weights are decimals
            sector_weights[sector] += float(data.get('weight', 0.0)) * 100.0
            print(f"  - {ticker_symbol}: {sector}")
        except Exception as e:
            print(f"Could not fetch sector for {ticker_symbol}: {e}")
            if 'Unknown' not in sector_weights:
                sector_weights['Unknown'] = 0.0
            sector_weights['Unknown'] += float(data.get('weight', 0.0)) * 100.0

    # Round the final percentages for cleanliness
    return {k: round(v, 2) for k, v in sector_weights.items()}


# Add this helper function inside engine.py

def _calculate_hhi(holdings: dict) -> float:
    """
    Calculates the HHI score for the portfolio to measure concentration.
    Weights are expected as decimals (e.g., 0.25 for 25%).
    """
    # HHI uses percentage points (e.g., 25, not 0.25)
    weights_as_percent = [float(data.get('weight', 0.0)) * 100.0 for data in holdings.values()]
    hhi = sum([(w ** 2) for w in weights_as_percent])
    return round(hhi, 2)



def analyze_portfolio(pi: PortfolioInput, options: Optional[AnalysisOptions] = None) -> Dict[str, Any]:
    options = options or AnalysisOptions()

    # Prepare a minimal holdings map for helper functions
    holdings_map: Dict[str, Dict[str, Any]] = {}
    for h in (pi.holdings or []):
        holdings_map[h.symbol.upper()] = {
            "weight": float(h.weight or 0.0),
            "name": (h.name or ""),
        }

    # 1. Calculate the new metrics using our helpers
    hhi_score = _calculate_hhi(holdings_map)
    sector_weights = _calculate_sector_weights(holdings_map)

    # 2. Identify missing key sectors for the LLM's context
    all_major_sectors = [
        "Technology", "Healthcare", "Financials", "Consumer Discretionary",
        "Communication Services", "Industrials", "Consumer Staples",
        "Energy", "Utilities", "Real Estate", "Materials"
    ]
    present_sectors = set(sector_weights.keys())
    missing_sectors = [s for s in all_major_sectors if s not in present_sectors]

    # 3. Construct the final, comprehensive data object
    analysis_result: Dict[str, Any] = {
        "portfolioHoldings": [
            {"ticker": sym, "companyName": info.get("name", ""), "weight": float(info.get("weight", 0.0)) * 100.0}
            for sym, info in holdings_map.items()
        ],
        "calculatedMetrics": {
            "hhiScore": hhi_score,
            "sectorWeights": sector_weights,
            "missingSectors": missing_sectors,
        },
    }

    return analysis_result
