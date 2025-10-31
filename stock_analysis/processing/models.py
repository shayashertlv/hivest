"""hivest/stock_analysis/processing/models.py"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


@dataclass
class StockInput:
    """Single-instrument analysis input."""
    symbol: str
    timeframe_label: str = "1y"   # e.g., "YTD", "6m", "1y"
    periods_per_year: float = 252.0

    # Time series (same periodicity)
    returns: List[float] = field(default_factory=list)
    benchmark_returns: Dict[str, List[float]] = field(default_factory=dict)  # e.g., {"SPY": [...], "QQQ": [...]}
    market_returns: Optional[List[float]] = None  # If None, will try SPY

    # Risk
    risk_free_rate: float = 0.0

    # Context
    upcoming_events: Optional[List[Dict[str, Any]]] = None


@dataclass
class AnalysisOptions:
    include_news: bool = True
    news_limit: int = 6
    include_events: bool = True
    verbose: bool = False


@dataclass
class ComputedMetrics:
    # Core perf
    cum_return: Optional[float] = None
    volatility: Optional[float] = None
    beta: Optional[float] = None
    sharpe: Optional[float] = None
    sortino: Optional[float] = None
    max_drawdown: Optional[float] = None

    # Benchmarks
    benchmarks: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)

    # Technicals (optional if data insufficient)
    rsi: Optional[float] = None
    sma20: Optional[float] = None
    sma50: Optional[float] = None
    sma200: Optional[float] = None
    pct_from_52w_high: Optional[float] = None
    pct_from_52w_low: Optional[float] = None

    # Spot price for sanity checks (optional)
    last_price: Optional[float] = None
    prev_close: Optional[float] = None  # Previous day's close
    daily_change: Optional[float] = None  # Last day's price change (fraction)
    daily_change_pct: Optional[float] = None  # Last day's price change (percentage)

    # Fundamentals and instrument typing
    instrument_type: str = "stock"  # "stock" | "etf" | "crypto" (best-effort)
    fundamentals: Dict[str, Any] = field(default_factory=dict)
    etf_profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StockAnalysisContext:
    stock_input: StockInput
    computed_metrics: ComputedMetrics
    news_items: List[Dict[str, Any]] = field(default_factory=list)
    upcoming_events: List[Dict[str, Any]] = field(default_factory=list)
