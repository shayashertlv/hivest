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
    dt_now = datetime.now().strftime("%Y-%-d %H:%M")

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
    # FIX: Added 'if v is not None' to prevent formatting errors
    fund = ", ".join(
        f"{k} ({fund_map.get(k, 'Metric')})={v:.2f}" for k, v in fund_items.items() if v is not None
    ) if fund_items else "none"
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

    # Add qualitative context to beta
    beta_val = metrics.beta_vs_spy
    beta_text = f"beta={beta_val:.2f}"
    if beta_val > 1.5:
        beta_text += " (significantly more volatile than the market)"
    elif beta_val > 1.1:
        beta_text += " (more volatile than the market)"
    elif beta_val < 0.9:
        beta_text += " (less volatile than the market)"
    else:
        beta_text += " (average market volatility)"

    perf = f"cum_return_1y={metrics.cum_return * 100:.2f}%; {beta_text}"
    tech = f"last_close={metrics.last_close:.2f}; rsi14={metrics.rsi14:.1f}; trend_50d_vs_200d={'Uptrend' if metrics.sma50 > metrics.sma200 else 'Downtrend'}"

    fund_items = metrics.fundamentals or {}

    # Add qualitative context to P/E Ratio
    pe_ratio = fund_items.get("peRatio")
    pe_text = ""
    if pe_ratio is not None:
        if pe_ratio > 50:
            pe_text = f"P/E Ratio={pe_ratio:.2f} (suggests high growth expectations or potential overvaluation)"
        elif pe_ratio > 25:
            pe_text = f"P/E Ratio={pe_ratio:.2f} (suggests a growth stock valuation)"
        else:
            pe_text = f"P/E Ratio={pe_ratio:.2f} (suggests a value stock valuation)"

    # Build a more descriptive fundamentals line
    fund_parts = [pe_text] if pe_text else []
    for k, v in fund_items.items():
        if k != "peRatio" and v is not None:
            fund_parts.append(f"{k}={v:.3g}")

    fund_line = ", ".join(fund_parts) if fund_parts else "none"

    news_block = _fmt_news(metrics.news_items, limit=3)
    market_news_block = _fmt_news(metrics.market_news_items, limit=3)

    instruction = (
        "You are a junior financial analyst. Your task is to analyze the raw data and extract key insights for a senior analyst. "
        "Focus on **interpreting** the numbers. Do NOT just repeat the data. "
        "Create a concise, bulleted list under the most relevant headings. Only create headings for which you have a meaningful insight.\n\n"
        "Possible Headings:\n"
        "- [Overall Performance]\n"
        "- [Key Strengths]\n"
        "- [Potential Risks]\n"
        "- [Valuation Insights]\n"
        "- [Technical Picture]\n"
        "- [Recent News & Catalysts]\n"
        "- [Market Outlook]"
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

    return f"{instruction}\n{context}\n--- OUTPUT (Bulleted list of insights only) ---"


# +++ REFACTOR STEP 2: Use the pre-analyzed insights to synthesize the final JSON +++
def build_json_synthesis_prompt(pre_analysis_output: str) -> str:
    """
    Builds the final prompt that takes pre-analyzed text and synthesizes it into a non-repetitive JSON.
    This is the second step in the new two-step process.
    """
    instruction = (
        "You are a senior financial analyst. You have been given a set of pre-analyzed points from your junior analyst. "
        "Your job is to synthesize these points into a clear, insightful, and **jargon-free** JSON report for a retail investor.\n\n"
        "**CRITICAL RULES:**\n"
        "1.  **NO FINANCIAL JARGON**: Do NOT use terms like 'beta', 'P/E ratio', 'RSI', 'SMA', or other technical indicators. Instead, **explain the conclusion** from those numbers in plain English. For example, instead of 'Beta is 1.8', say 'The stock is highly sensitive to market swings.'\n"
        "2.  **DYNAMIC STRUCTURE**: Do NOT use a fixed template. Build a JSON object where the keys are descriptive titles for the insights you provide. Only include a key if you have a meaningful, non-generic insight for it.\n"
        "3.  **SINGLE STRING VALUES**: The value for each key in the JSON object must be a **single string**. Combine all related points into one coherent paragraph. **Do not use lists or arrays of strings.**\n"
        "4.  **BE SPECIFIC AND CONCISE**: Base your entire output **exclusively** on the provided points. If the points mention a partnership with a specific company, name that company. If they don't, do not mention partnerships at all.\n"
        "5.  **SYNTHESIZE, DON'T REPEAT**: Combine related points into a coherent narrative under a single key. Each core idea should appear only once.\n"
        "6.  **EXAMPLE KEYS**: You might use keys like `summary`, `positiveOutlook`, `potentialRisks`, `valuationCommentary`, `recentDevelopments`, or `stockPerformance`, but you should choose the most appropriate keys based on the provided text.\n\n"
        "Your final output must be **only the JSON object** and nothing else."
    )

    context = f"--- PRE-ANALYZED POINTS FROM JUNIOR ANALYST ---\n{pre_analysis_output}"

    return (
        f"{instruction}\n\n"
        f"{context}\n\n"
        f"--- OUTPUT (Valid, jargon-free JSON with single string values only) ---"
    )