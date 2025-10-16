"""hivest/__init__.py"""
"""
Hivest: Portfolio and stock analysis pipelines with a shared data/LLM layer.

Public API re-exports:
- Portfolio: Holding, PortfolioInput, AnalysisOptions, ComputedMetrics, analyze_portfolio, holdings_from_user_positions
- Stock: StockInput, AnalysisOptions, ComputedMetrics, StockAnalysisContext, analyze_stock
"""

# Portfolio
from .portfolio_analysis.processing.models import Holding, PortfolioInput, AnalysisOptions, ComputedMetrics
from .portfolio_analysis.processing.engine import analyze_portfolio, holdings_from_user_positions
from .portfolio_analysis.llm.prompts import build_portfolio_prompt

# Stock
from .stock_analysis.processing.models import StockInput, AnalysisOptions as StockAnalysisOptions, ComputedMetrics as StockComputedMetrics, StockAnalysisContext
from .stock_analysis.processing.engine import analyze_stock
