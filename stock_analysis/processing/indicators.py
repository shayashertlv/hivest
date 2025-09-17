from typing import List

def sma(xs: List[float], n: int) -> float:
    if not xs or n <= 0 or len(xs) < n: return 0.0
    return sum(xs[-n:]) / n

def compute_rsi(closes: List[float], period: int = 14) -> float:
    if not closes or len(closes) < period + 1: return 50.0
    gains, losses = [], []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i-1]
        (gains if ch > 0 else losses).append(abs(ch))
    avg_gain = sum(gains[-period:]) / max(1, period)
    avg_loss = sum(losses[-period:]) / max(1, period)
    if avg_loss == 0: return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))

def pct_from_window_extrema(closes: List[float], window: int = 252):
    if not closes: return 0.0, 0.0
    w = closes[-window:] if len(closes) >= window else closes
    hi, lo, last = max(w), min(w), closes[-1]
    pct_from_hi = (last / hi - 1.0) if hi else 0.0
    pct_from_lo = (last / lo - 1.0) if lo else 0.0
    return pct_from_hi, pct_from_lo
