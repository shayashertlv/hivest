from datetime import datetime
from ..processing.models import StockMetrics

def build_stock_prompt(symbol: str, tf_label: str, m: StockMetrics) -> str:
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    perf = f"{tf_label}: cum={m.cum_return*100:.2f}%, vol={m.volatility*100:.2f}%, beta={m.beta_vs_spy:.2f}, maxDD={m.max_drawdown*100:.2f}%."
    tech = f"RSI14={m.rsi14:.1f}; SMA20={m.sma20:.2f}; SMA50={m.sma50:.2f}; SMA200={m.sma200:.2f}; off_52w_high={m.pct_from_52w_high*100:.2f}%; off_52w_low={m.pct_from_52w_low*100:.2f}%."
    fund = ", ".join(f"{k}={v:.2f}" for k,v in m.fundamentals.items()) if m.fundamentals else "n/a"
    news = "\n".join(f"- {it.get('title','')} [source={it.get('source','')}; date={(it.get('publishedAt','') or '')[:10]}]" for it in (m.news_items or [])[:4]) or "none"
    nxt = m.next_earnings or "none"

    instruction = (
      "Write one cohesive paragraph for a retail investor. "
      "Start with numeric change (return vs risk/beta), then technical posture, "
      "then valuation context if present, then upcoming catalysts, then a balanced suggestion (hold/trim/add) with two watch items. "
      "Plain English. No bullets. Do not invent numbers."
    )
    payload = (
      f"Generated: {dt_now}\n"
      f"Symbol: {symbol.upper()}\n"
      f"Performance: {perf}\n"
      f"Technicals: {tech}\n"
      f"Fundamentals: {fund}\n"
      f"NextEarnings: {nxt}\n"
      f"News:\n{news}\n"
    )
    return instruction + "\n\n--- DATA ---\n" + payload
