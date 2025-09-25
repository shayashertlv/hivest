from datetime import datetime
from ..processing.models import StockMetrics


def build_stock_prompt(symbol: str, tf_label: str, m: StockMetrics) -> str:
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    perf = f"{tf_label}: cum={m.cum_return*100:.2f}%, vol={m.volatility*100:.2f}%, beta={m.beta_vs_spy:.2f}, maxDD={m.max_drawdown*100:.2f}%."
    tech = f"RSI14={m.rsi14:.1f}; SMA20={m.sma20:.2f}; SMA50={m.sma50:.2f}; SMA200={m.sma200:.2f}; off_52w_high={m.pct_from_52w_high*100:.2f}%; off_52w_low={m.pct_from_52w_low*100:.2f}%."
    fund = ", ".join(f"{k}={v:.2f}" for k, v in m.fundamentals.items()) if m.fundamentals else "n/a"
    news = "\n".join(
        f"- {it.get('title','')} [source={it.get('source','')}; date={(it.get('publishedAt','') or '')[:10]}]"
        for it in (m.news_items or [])[:4]
    ) or "none"
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


def build_stock_json_prompt(symbol: str, metrics: StockMetrics) -> str:
    """
    Builds a prompt to generate an interpretive, narrative-focused stock analysis.
    """
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    inst = (metrics.instrument_type or "stock").lower().strip()

    # --- Contextual Data for the LLM ---
    perf = f"cum_return_1y={metrics.cum_return * 100:.2f}%; beta={metrics.beta_vs_spy:.2f}"
    tech = f"last_close={metrics.last_close:.2f}; rsi14={metrics.rsi14:.1f}; support_52w_low={metrics.low_52w:.2f}; resistance_52w_high={metrics.high_52w:.2f}"

    fund_pairs = [f"{k}={v:.3g}" for k, v in (metrics.fundamentals or {}).items() if v is not None]
    fund_line = ", ".join(fund_pairs) if fund_pairs else "none"

    news_block = "\n".join([f"- {item.get('title', '')}" for item in (metrics.news_items or [])[:3]]) or "none"
    market_news_block = "\n".join(
        [f"- {item.get('title', '')}" for item in (metrics.market_news_items or [])[:3]]) or "none"

    # Pass the raw score to the LLM for better nuance
    social_sentiment_score = "not available"
    if metrics.social_sentiment and isinstance(metrics.social_sentiment, dict) and metrics.social_sentiment.get(
            'sentimentScore'):
        social_sentiment_score = f"{metrics.social_sentiment.get('sentimentScore'):.2f}"

    # --- The New JSON Schema with Interpretive Instructions ---
    schema = (
        '{\n'
        '  "introduction": "string (A 2-3 sentence intro. Start with what the company is, then its performance over the last year, and finally a note on its latest business focus. **Avoid technical jargon like \'beta\'.**)",\n'
        '  "analystRating": {\n'
        '    "rating": "string (e.g., \'Buy\', \'Hold\')",\n'
        '    "rationale": "string (Explain the rating. Reference a specific piece of news and explain WHY it supports the long-term potential.)"\n'
        '  },\n'
        '  "keyDrivers": {\n'
        '    "strength": "string (**Interpret the fundamental data.** Example: \'The company shows strong profitability with an impressive 25% growth in earnings, indicating high demand.\' Do NOT just state the raw number.)",\n'
        '    "concern": "string (Describe the main risk or concern.)",\n'
        '    "opportunity": "string (Describe the biggest growth opportunity.)"\n'
        '  },\n'
        '  "financialsAndTechnicals": "string (A 2-sentence summary. First, summarize the most important fundamental facts. Second, **provide conclusions from the technicals.** Example: \'Technicals suggest the stock has strong upward momentum but may be approaching overbought territory.\' Do NOT state the raw RSI or price levels.)",\n'
        '  "sentiment": {\n'
        '    "news": "string (Summarize the news sentiment. **Provide nuance.** Example: \'News sentiment is cautiously optimistic, focusing on long-term potential while noting short-term risks.\')",\n'
        '    "social": "string (Summarize the public\'s sentiment from social media. **Provide nuance.** Example: \'Social media sentiment is largely positive, with retail investors expressing excitement about the upcoming product launch.\')"\n'
        '  },\n'
        '  "projections": {\n'
        '    "nextWeek": "string (A qualitative forecast for the next week, **basing the reasoning on both the company-specific news and the broader market news provided.**)",\n'
        '    "sixMonths": "string (A price target or range based on analyst estimates.)",\n'
        '    "oneYear": "string (A price target or range based on analyst estimates.)"\n'
        '  }\n'
        '}'
    )

    # --- The New Instructions for the LLM ---
    instruction = (
        "You are a financial analyst writing a report for a retail investor. Your tone is professional and easy to understand. "
        "Your task is to generate a single, clean JSON object with no extra text or markdown.\n"
        "Follow the provided schema exactly. **Your primary goal is to interpret the data and provide conclusions, not to repeat the raw numbers.**"
    )

    context = (
        f"--- DATA FOR {symbol.upper()} ---\n"
        f"Date: {dt_now}\n"
        f"Performance: {perf}\n"
        f"Technicals: {tech}\n"
        f"Fundamentals: {fund_line}\n"
        f"Social Sentiment Score (from -1 to 1): {social_sentiment_score}\n"
        f"Recent News for {symbol.upper()}:\n{news_block}\n"
        f"Recent General Market News (SPY):\n{market_news_block}\n"
    )

    return f"{instruction}\n\n--- REQUIRED JSON SCHEMA ---\n{schema}\n\n{context}\n\n--- OUTPUT (JSON only) ---"