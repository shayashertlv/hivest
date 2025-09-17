from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple


@dataclass
class Holding:
    symbol: str
    weight: float  # fraction of portfolio, not percent
    avg_buy_price: Optional[float] = None  # average buying price per share
    bought_at: Optional[str] = None        # purchase date (YYYY-MM-DD) if known
    sector: Optional[str] = None
    name: Optional[str] = None


@dataclass
class PortfolioInput:
    # Core composition
    holdings: List[Holding]

    # Time horizon and cadence
    timeframe_label: str = "YTD"  # Always analysed on a YTD window
    periods_per_year: float = 252.0     # daily default

    # Performance series (same periodicity)
    portfolio_returns: List[float] = field(default_factory=list)
    benchmark_returns: Dict[str, List[float]] = field(default_factory=dict)  # e.g., {"SPY": [...], "QQQ": [...]} 
    market_returns: Optional[List[float]] = None  # for beta; if None, will try SPY
    per_symbol_returns: Optional[Dict[str, List[float]]] = None  # optional per-position series

    # Risk settings
    risk_free_rate: float = 0.0  # in the same periodicity as returns

    # Events and context
    upcoming_events: Optional[List[Dict]] = None  # e.g., [{symbol,type,date,note}]


@dataclass
class AnalysisOptions:
    top_k_winners_losers: int = 5
    concentration_threshold: float = 0.1
    include_news: bool = True
    news_limit: int = 8
    include_events: bool = True
    verbose: bool = False           # print progress while processing
    echo_tools: bool = True         # print which data sources / tools were used
    deliberate_passes: int = 2        # 0/1 = direct; 2 = plan→polish
    insight_mode: str = "rules+llm"   # "none" | "rules" | "rules+llm"
    llm_model: Optional[str] = None
    llm_host: Optional[str] = None


@dataclass
class ComputedMetrics:
    cum_return: float
    benchmarks: Dict[str, Dict]  # from benchmark_comparison
    by_symbol: List[Tuple[str, float]]
    winners: List[Tuple[str, float]]
    losers: List[Tuple[str, float]]
    by_sector: List[Tuple[str, float]]
    top_sectors: List[Tuple[str, float]]
    bottom_sectors: List[Tuple[str, float]]
    volatility: float
    beta: float
    sharpe: float
    concentration: Dict
    max_drawdown: float = 0.0
    calmar_like: float = 0.0
    sortino: float = 0.0
    volatility_annual: float = 0.0
    sharpe_annual: float = 0.0
    sortino_annual: float = 0.0
