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
    schema = """{
      "introduction": "string (A 2-3 sentence intro. Start with what the company is, then its performance over the last year, and finally a note on its latest business focus. **Avoid technical jargon like 'beta'.**)",
      "analystRating": {
        "rating": "string (e.g., 'Buy', 'Hold')",
        "rationale": "string (Explain the rating. Reference a specific piece of news and explain WHY it supports the long-term potential.)"
      },
      "keyDrivers": {
        "strength": "string (**Interpret the fundamental data.** Example: 'The company shows strong profitability with an impressive earnings trajectory, indicating high demand.' Do NOT just state raw numbers.)",
        "concern": "string (Describe the main risk or concern.)",
        "opportunity": "string (Describe the biggest growth opportunity.)"
      },
      "financialsAndTechnicals": "string (A 2-sentence summary. First, summarize the most important fundamental facts. Second, **provide conclusions from the technicals.** Example: 'Technicals suggest the stock has strong upward momentum but may be approaching overbought territory.' Do NOT state the raw RSI or price levels.)",
      "financialTrends": {
        "servicesGrowth": "string (Interpret the Services growth trend and why it matters for margins, stability, or valuation.)",
        "buybacks": "string (Summarize capital returns via buybacks and their impact on EPS durability or downside support—interpretive, not numeric.)",
        "revenueMix": "string (Describe notable shifts in mix—e.g., iPhone vs. Services/Wearables—and what those shifts imply for growth and resilience.)"
      },
      "peerComparison": {
        "peerSet": "string (Name the key peers used for comparison; e.g., 'MSFT, GOOGL, META'.)",
        "relativeStrengths": "string (Where the company is stronger than peers—ecosystem lock-in, margins, platform advantages—interpretive only.)",
        "relativeWeaknesses": "string (Where it lags peers—valuation premium, slower segment growth, dependency risks—interpretive only.)"
      },
      "riskSurface": {
        "competition": "string (Competitive threats and pricing pressure from direct and adjacent players.)",
        "execution": "string (Execution risks like product timing, adoption curves, integration, or channel.)",
        "macroAndRegulatory": "string (Macro, FX, supply-chain, and regulatory/antitrust exposure with brief implications.)",
        "mitigants": "string (Key factors that could offset these risks—balance sheet strength, loyal user base, ecosystem, or diversification.)"
      },
      "sentiment": {
        "news": "string (Summarize the news sentiment. **Provide nuance.** Example: 'News sentiment is cautiously optimistic, focusing on long-term potential while noting short-term risks.')",
        "social": "string (Summarize the public's sentiment from social media. **Provide nuance.** Example: 'Social media sentiment is largely positive, with retail investors expressing excitement about the upcoming product launch.')"
      },
      "projections": {
        "nextWeek": "string (A qualitative forecast for the next week, **basing the reasoning on both the company-specific news and the broader market news provided.**)",
        "sixMonths": "string (A price target or range based on analyst estimates.)",
        "oneYear": "string (A price target or range based on analyst estimates.)"
      }
    }"""

    # --- The New Instructions for the LLM ---
    instruction = (
        "You are a financial analyst writing a report for a retail investor. Your tone is professional and easy to understand. "
        "Your task is to generate a single, clean JSON object with no extra text or markdown. "
        "Follow the provided schema exactly. **Your primary goal is to interpret the data and provide conclusions, not to repeat raw numbers.** "
        "For technicals, do not state specific indicators or price levels—only conclusions. "
        "For peerComparison, discuss relative positioning versus named peers without quoting raw multiples—focus on why the company is stronger or weaker. "
        "For financialTrends, interpret Services growth, buybacks, and revenue mix shifts (e.g., iPhone vs. Services/Wearables) and explain why they matter. "
        "For riskSurface, cover competition, execution, macro/regulatory, and note any mitigants. "
        "Be concise, concrete, and avoid jargon where possible."
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