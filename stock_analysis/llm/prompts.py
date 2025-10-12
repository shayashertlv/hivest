# prompts.py

from datetime import datetime
from typing import List, Dict, Any
from ..processing.models import StockMetrics


# -------- helpers (internal - no changes needed) --------
def _dedup_by_title(items: List[Dict[str, Any]] | None) -> List[Dict[str, Any]]:
    """Remove near-duplicate news by normalized title."""
    seen = set()
    out = []
    for it in (items or []):
        t = (it.get("title") or "").strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(it)
    return out


def _fmt_news(items: List[Dict[str, Any]] | None, limit: int = 3) -> str:
    """Format news as '- title (Source, YYYY-MM-DD)' lines."""
    items = _dedup_by_title(items)[:limit]
    if not items:
        return "none"
    lines = []
    for it in items:
        title = (it.get("title") or "").strip()
        src = (it.get("source") or "").strip()
        date = ((it.get("publishedAt") or "")[:10]).strip()
        tag = ", ".join(x for x in (src or None, date or None) if x) or ""
        lines.append(f"- {title}" + (f" ({tag})" if tag else ""))
    return "\n".join(lines)


# -------- 1) short natural-language paragraph (Revised) --------
def build_stock_prompt(symbol: str, tf_label: str, m: StockMetrics) -> str:
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    perf = (
        f"{tf_label}: cum={m.cum_return * 100:.2f}%, vol={m.volatility * 100:.2f}%, "
        f"beta={m.beta_vs_spy:.2f}, maxDD={m.max_drawdown * 100:.2f}%."
    )
    tech = (
        f"RSI14={m.rsi14:.1f}; SMA20={m.sma20:.2f}; SMA50={m.sma50:.2f}; SMA200={m.sma200:.2f}; "
        f"off_52w_high={m.pct_from_52w_high * 100:.2f}%; off_52w_low={m.pct_from_52w_low * 100:.2f}%."
    )
    # +++ Added context tags to fundamentals for better model interpretation +++
    fund_items = m.fundamentals or {}
    fund_map = {
        "peRatio": "Valuation", "roe": "Quality", "debtToEquity": "Balance Sheet",
        "cr": "Liquidity", "payoutRatio": "Shareholder Return"
    }
    fund = ", ".join(f"{k} ({fund_map.get(k, 'Metric')})={v:.2f}" for k, v in fund_items.items()) if fund_items else "none"
    nxt = m.next_earnings or "none"
    news = _fmt_news(m.news_items, limit=3)

    # +++ Simplified and sharpened instructions for better, more natural prose +++
    instruction = (
        "You are a sharp financial analyst. Write a dense, 90-120 word paragraph summarizing the stock for a retail investor. "
        "Guiding Principles:\n"
        "1. BE AN ANALYST, NOT A DATA REPORTER: Interpret the data to tell a story. Do not echo raw numbers like 'RSI is 71'. Instead, say 'momentum appears stretched'.\n"
        "2. VARY YOUR LANGUAGE: Avoid repetitive sentence structures and clichés.\n"
        "3. BE CONCLUSIVE: End with a single, clear, actionable takeaway (e.g., 'ideal for patient investors' or 'wait for a pullback before entering').\n"
        "Your summary must synthesize its performance, a key technical takeaway, a core fundamental strength/weakness, and one recent news catalyst."
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


# -------- 2) structured JSON report (Heavily Revised) --------
def build_stock_json_prompt(symbol: str, metrics: StockMetrics) -> str:
    """
    Builds a prompt that yields a clean, non-repetitive interpretive JSON.
    """
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    inst = (metrics.instrument_type or "stock").lower().strip()

    perf = f"cum_return_1y={metrics.cum_return * 100:.2f}%; beta={metrics.beta_vs_spy:.2f}"
    tech = f"last_close={metrics.last_close:.2f}; rsi14={metrics.rsi14:.1f}; trend_50d_vs_200d={'Uptrend' if metrics.sma50 > metrics.sma200 else 'Downtrend'}"

    # +++ Added context tags to fundamentals for better model interpretation +++
    fund_items = metrics.fundamentals or {}
    fund_map = {
        "peRatio": "Valuation", "roe": "Quality", "debtToEquity": "Balance Sheet",
        "cr": "Liquidity", "payoutRatio": "Shareholder Return"
    }
    fund_line = ", ".join(f"{k} ({fund_map.get(k, 'Metric')})={v:.3g}" for k, v in fund_items.items() if v is not None) if fund_items else "none"

    news_block = _fmt_news(metrics.news_items, limit=3)
    market_news_block = _fmt_news(metrics.market_news_items, limit=3)

    # +++ Simplified schema to be more robust and less prone to repetition +++
    schema = """{
      "executiveSummary": {
        "rating": "string ('Buy'/'Hold'/'Sell')",
        "priceTarget12M": "string (e.g., '$290' or 'N/A')",
        "summary": "string (2-3 sentences. The 'bottom line up front'. What is the core thesis for the rating?)"
      },
      "investmentThesis": [
        "string (A bullet point explaining a core strength or opportunity. E.g., 'Dominant Market Position: The company's ecosystem creates high switching costs and ensures customer loyalty.')",
        "string (A second, distinct bullet point.)"
      ],
      "keyRisks": [
        "string (A bullet point explaining a core weakness or threat. E.g., 'Regulatory Scrutiny: Increased government oversight of app store policies could impact high-margin service revenues.')",
        "string (A second, distinct bullet point.)"
      ],
      "valuationAndTechnicals": {
        "valuation": "string (1-2 sentences. Is it expensive or cheap vs. peers/history, and why? Interpret the fundamental data provided.)",
        "technicals": "string (1 sentence. What is the key takeaway from the technical data? E.g., 'The stock is in a firm long-term uptrend but appears overbought in the short-term, suggesting a pullback is possible.')"
      },
      "outlookAndSentiment": {
        "catalysts": "string (1 sentence on near-term catalysts, citing a specific news item if relevant.)",
        "marketContext": "string (1 sentence synthesizing general market news tone and its potential impact.)"
      }
    }"""

    # +++ Sharpened instructions with core principles +++
    instruction = (
        "You are a concise financial analyst for a retail audience. Generate a single, valid JSON object based on the schema and data. Follow these principles:\n"
        "1. BE AN INSIGHTFUL ANALYST, NOT A DATA REPORTER: Interpret the data, don't just repeat it. Connect the dots between fundamentals, news, and the stock's story.\n"
        "2. SYNTHESIZE, DON'T REPEAT: Each field must offer a unique insight. Do not state the same core idea (e.g., 'strong brand') in multiple sections.\n"
        "3. QUANTIFY TO JUSTIFY: Back up a claim with a key piece of data, but don't just list numbers. Example: 'Valuation appears stretched, trading at a premium to its historical average.'\n"
        "4. BE CLEAR AND CONCISE: Use plain English. No jargon. No filler."
    )

    context = (
        f"--- DATA FOR {symbol.upper()} ({inst}) ---\n"
        f"Date: {dt_now}\n"
        f"Performance: {perf}\n"
        f"Technicals: {tech}\n"
        f"Fundamentals: {fund_line}\n"
        f"NextEarnings: {metrics.next_earnings or 'none'}\n"
        f"Recent News for {symbol.upper()}:\n{news_block}\n"
        f"Recent General Market News:\n{market_news_block}\n"
    )

    return (
        f"{instruction}\n\n"
        f"--- REQUIRED JSON SCHEMA ---\n{schema}\n\n"
        f"{context}\n"
        f"--- OUTPUT (JSON only) ---"
    )