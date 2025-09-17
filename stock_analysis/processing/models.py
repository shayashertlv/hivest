from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class StockInput:
    symbol: str
    timeframe_label: str = "6m"
    periods_per_year: float = 252.0
    avg_buy_price: Optional[float] = None
    bought_at: Optional[str] = None  # YYYY-MM-DD

@dataclass
class StockOptions:
    include_news: bool = True
    include_events: bool = True
    news_limit: int = 8
    verbose: bool = False
    llm_model: Optional[str] = None
    llm_host: Optional[str] = None
    echo_tools: bool = True

@dataclass
class StockMetrics:
    dates: List[str] = field(default_factory=list)
    closes: List[float] = field(default_factory=list)
    returns: List[float] = field(default_factory=list)
    cum_return: float = 0.0
    volatility: float = 0.0
    beta_vs_spy: float = 0.0
    max_drawdown: float = 0.0
    rsi14: float = 0.0
    sma20: float = 0.0
    sma50: float = 0.0
    sma200: float = 0.0
    pct_from_52w_high: float = 0.0
    pct_from_52w_low: float = 0.0
    fundamentals: Dict[str, float] = field(default_factory=dict)
    news_items: List[Dict] = field(default_factory=list)
    next_earnings: Optional[str] = None

@dataclass
class StockReport:
    metrics: StockMetrics
    narrative: str
