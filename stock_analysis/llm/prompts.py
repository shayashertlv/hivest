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


# -------- 1) short natural-language paragraph (Minor Polish) --------
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
    fund_items = m.fundamentals or {}
    fund_map = {
        "peRatio": "Valuation", "roe": "Quality", "debtToEquity": "Balance Sheet",
        "cr": "Liquidity", "payoutRatio": "Shareholder Return"
    }
    fund = ", ".join(
        f"{k} ({fund_map.get(k, 'Metric')})={v:.2f}" for k, v in fund_items.items()) if fund_items else "none"
    nxt = m.next_earnings or "none"
    news = _fmt_news(m.news_items, limit=3)

    # +++ Polished persona and instructions for more consistent tone +++
    instruction = (
        "You are a sharp, insightful financial analyst writing for a smart retail investor audience. Your tone is professional yet direct.\n"
        "Generate a dense, 90-120 word paragraph synthesizing the provided data. Adhere to these principles:\n"
        "1. INTERPRET, DON'T RECITE: Transform raw numbers into insights. Instead of 'RSI is 71', say 'the stock shows strong short-term momentum but may be overbought'.\n"
        "2. VARY LANGUAGE: Avoid repetitive sentence structures.\n"
        "3. BE CONCLUSIVE: End with a clear, actionable takeaway (e.g., 'This appears suitable for value investors' or 'Traders might wait for a pullback').\n"
        "Your summary must weave together its performance, a key technical observation, a core fundamental insight, and a relevant news catalyst."
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


# -------- 2) structured JSON report (Major Refactor to Two-Step Process) --------

# +++ REFACTOR STEP 1: Create a pre-analysis prompt to extract thematic insights first +++
def build_pre_analysis_prompt(symbol: str, metrics: StockMetrics) -> str:
    """
    Builds a prompt that interprets raw data into thematic bullet points.
    This is the first step in the new two-step JSON generation process.
    """
    inst = (metrics.instrument_type or "stock").lower().strip()
    perf = f"cum_return_1y={metrics.cum_return * 100:.2f}%; beta={metrics.beta_vs_spy:.2f}"
    tech = f"last_close={metrics.last_close:.2f}; rsi14={metrics.rsi14:.1f}; trend_50d_vs_200d={'Uptrend' if metrics.sma50 > metrics.sma200 else 'Downtrend'}"

    fund_items = metrics.fundamentals or {}
    fund_map = {
        "peRatio": "Valuation (TTM)", "roe": "Quality", "debtToEquity": "Balance Sheet",
        "cr": "Liquidity", "payoutRatio": "Shareholder Return",
        "analyst_target_price": "Analyst Avg Target Price"
    }
    fund_line = ", ".join(f"{k} ({fund_map.get(k, 'Metric')})={v:.3g}" for k, v in fund_items.items() if
                          v is not None) if fund_items else "none"

    news_block = _fmt_news(metrics.news_items, limit=3)
    market_news_block = _fmt_news(metrics.market_news_items, limit=3)

    instruction = (
        "You are a junior financial analyst. Your task is to analyze the raw data provided for the stock and extract key interpretive insights. Do NOT write a final report or JSON. "
        "Instead, create a concise, bulleted list under the following specific headings. Each bullet must be a distinct idea derived directly from the data.\n\n"
        "Headings to use:\n"
        "- [Strengths & Opportunities]\n"
        "- [Weaknesses & Risks]\n"
        "- [Valuation Insights]\n"
        "- [Technical Insights]\n"
        "- [Catalysts & Market Context]\n"
    )

    context = (
        f"--- DATA FOR {symbol.upper()} ({inst}) ---\n"
        f"Performance: {perf}\n"
        f"Technicals: {tech}\n"
        f"Fundamentals: {fund_line}\n"
        f"NextEarnings: {metrics.next_earnings or 'none'}\n"
        f"Recent Company News:\n{news_block}\n"
        f"Recent Market News:\n{market_news_block}\n"
    )

    return f"{instruction}\n{context}\n--- OUTPUT (Bulleted list only) ---"


# +++ REFACTOR STEP 2: Use the pre-analyzed insights to synthesize the final JSON +++
def build_json_synthesis_prompt(pre_analysis_output: str) -> str:
    """
    Builds the final prompt that takes pre-analyzed text and synthesizes it into a non-repetitive JSON.
    This is the second step in the new two-step process.
    """
    schema = """{
      "executiveSummary": {
        "rating": "string ('Buy'/'Hold'/'Sell' or 'N/A')",
        "priceTarget12M": "string (e.g., '$290' or 'N/A')",
        "summary": "string (2-3 sentences. The 'bottom line up front'. What is the core thesis for the rating?)"
      },
      "investmentThesis": ["string (A bullet point explaining a core strength or opportunity.)", "string (A second, distinct bullet point.)"],
      "keyRisks": ["string (A bullet point explaining a core weakness or threat.)", "string (A second, distinct bullet point.)"],
      "valuationAndTechnicals": {
        "valuation": "string (1-2 sentences interpreting the valuation insights.)",
        "technicals": "string (1 sentence interpreting the technical insights.)"
      },
      "outlookAndSentiment": {
        "catalysts": "string (1 sentence on near-term catalysts.)",
        "marketContext": "string (1 sentence on the market context.)"
      }
    }"""

    instruction = (
        "You are a senior financial analyst. You have been given a set of pre-analyzed points from your junior analyst. Your job is to synthesize these points into a polished, final JSON report according to the provided schema.\n\n"
        "**CRITICAL RULES:**\n"
        "1. USE ONLY THE PRE-ANALYZED POINTS: Your entire JSON output must be based exclusively on the text in the '--- PRE-ANALYZED POINTS ---' block.\n"
        "2. NO REPETITION: Each core idea from the points must be mentioned ONLY ONCE in the entire JSON. Assign each point to its single most appropriate field.\n"
        "3. FILL THE SCHEMA: Populate all fields in the schema. If a point is not available for a field (e.g., no target price was analyzed), use 'N/A'."
    )

    context = f"--- PRE-ANALYZED POINTS ---\n{pre_analysis_output}"

    return (
        f"{instruction}\n\n"
        f"--- REQUIRED JSON SCHEMA ---\n{schema}\n\n"
        f"{context}\n\n"
        f"--- OUTPUT (Valid JSON only) ---"
    )