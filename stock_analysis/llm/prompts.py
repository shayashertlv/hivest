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
    Builds a prompt to generate a narrative-focused, readable stock analysis.
    """
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    inst = (metrics.instrument_type or "stock").lower().strip()

    # --- Contextual Data for the LLM ---
    perf = f"cum_return_1y={metrics.cum_return*100:.2f}%; beta={metrics.beta_vs_spy:.2f}"
    tech = f"last_close={metrics.last_close:.2f}; rsi14={metrics.rsi14:.1f}; support_52w_low={metrics.low_52w:.2f}; resistance_52w_high={metrics.high_52w:.2f}"
    
    fund_pairs = [f"{k}={v:.3g}" for k, v in (metrics.fundamentals or {}).items() if v is not None]
    fund_line = ", ".join(fund_pairs) if fund_pairs else "none"

    news_block = "\n".join([f"- {item.get('title', '')}" for item in (metrics.news_items or [])[:3]]) or "none"
    
    social_sentiment_summary = "neutral"
    if metrics.social_sentiment and isinstance(metrics.social_sentiment, dict) and metrics.social_sentiment.get('sentimentScore'):
        score = metrics.social_sentiment.get('sentimentScore', 0)
        if score > 0.5: social_sentiment_summary = "positive"
        elif score < -0.5: social_sentiment_summary = "negative"

    # --- The New JSON Schema ---
    schema = (
        '{\n'
        '  "introduction": "string (A 2-3 sentence intro. Start with what the company is, then its performance over the last year, and finally a note on its latest business focus based on recent news.)",\n'
        '  "analystRating": {\n'
        '    "rating": "string (e.g., \'Buy\', \'Hold\')",\n'
        '    "rationale": "string (Explain the rating. Reference a specific piece of news and explain WHY it supports the long-term potential.)"\n'
        '  },\n'
        '  "keyDrivers": {\n'
        '    "strength": "string (Elaborate on the company\'s primary strength in more detail, citing a fundamental metric.)",\n'
        '    "concern": "string (Describe the main risk or concern.)",\n'
        '    "opportunity": "string (Describe the biggest growth opportunity.)"\n'
        '  },\n'
        '  "financialsAndTechnicals": "string (A 2-sentence summary. First, summarize the most important fundamental facts like profitability and valuation. Second, summarize the key technicals like momentum and price levels.)",\n'
        '  "sentiment": {\n'
        '    "news": "string (A one-sentence summary of the overall sentiment from recent news headlines.)",\n'
        '    "social": "string (A one-sentence summary reflecting the public\'s sentiment based on the provided social media data.)"\n'
        '  },\n'
        '  "projections": {\n'
        '    "nextWeek": "string (A qualitative forecast for the next week based on current momentum, e.g., \'Likely to continue its upward trend.\') ",\n'
        '    "nextMonth": "string (A qualitative forecast for the next month.)",\n'
        '    "sixMonths": "string (A price target or range based on analyst estimates.)",\n'
        '    "oneYear": "string (A price target or range based on analyst estimates.)"\n'
        '  }\n'
        '}'
    )

    # --- The New Instructions for the LLM ---
    instruction = (
        "You are a financial analyst writing a report for a retail investor. Your tone is professional, insightful, and easy to understand. "
        "Your task is to generate a single, clean JSON object with no extra text or markdown.\n"
        "Follow the provided schema exactly. Do not add or remove keys.\n"
        "Base your entire analysis on the data provided below. Do not invent facts or numbers."
    )

    context = (
        f"--- DATA FOR {symbol.upper()} ---\n"
        f"Date: {dt_now}\n"
        f"Instrument Type: {inst}\n"
        f"Performance: {perf}\n"
        f"Technicals: {tech}\n"
        f"Fundamentals: {fund_line}\n"
        f"Social Sentiment Score: {social_sentiment_summary}\n"
        f"Recent News Headlines:\n{news_block}\n"
    )

    return f"{instruction}\n\n--- REQUIRED JSON SCHEMA ---\n{schema}\n\n{context}\n\n--- OUTPUT (JSON only) ---"
