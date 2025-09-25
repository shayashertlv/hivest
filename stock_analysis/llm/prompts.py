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
    Build a strict instruction prompt that forces the model to return ONLY a raw JSON object
    following the specified schema. Injects numeric/contextual data from StockMetrics to guide
    the model's filling of fields. Absolutely forbid any extra prose or Markdown.
    """
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    inst = (metrics.instrument_type or "stock").lower().strip()

    # Compact context lines for the model to ground on real numbers
    tech = (
        f"RSI14={metrics.rsi14:.1f}; "
        f"last_close={metrics.last_close:.2f}; support1={metrics.low_52w:.2f}; resistance1={metrics.high_52w:.2f}; support2_sma50={metrics.sma50:.2f}; "
        f"pct_from_52w_high={metrics.pct_from_52w_high*100:.2f}%; pct_from_52w_low={metrics.pct_from_52w_low*100:.2f}%"
    )
    perf = (
        f"cum_return={metrics.cum_return*100:.2f}%; volatility={metrics.volatility*100:.2f}%; "
        f"beta_vs_spy={metrics.beta_vs_spy:.2f}; max_drawdown={metrics.max_drawdown*100:.2f}%"
    )
    fund_pairs = []
    for k, v in (metrics.fundamentals or {}).items():
        try:
            fund_pairs.append(f"{k}={float(v):.4g}")
        except Exception:
            fund_pairs.append(f"{k}={v}")
    fund_line = ", ".join(fund_pairs) if fund_pairs else "none"

    # If ETF, inject expense ratio into fundamentals line
    if inst == "etf" and getattr(metrics, "etf_profile", None) is not None:
        er = getattr(metrics.etf_profile, "expense_ratio", None)
        if er is not None:
            er_str = f"expense_ratio={float(er):.4g}"
            fund_line = er_str if fund_line == "none" else f"{fund_line}, {er_str}"

    # Prepare Top 5 holdings block for ETFs
    holdings_block = None
    if inst == "etf" and getattr(metrics, "etf_profile", None) is not None:
        top = list(getattr(metrics.etf_profile, "top_holdings", []) or [])
        lines = []
        for it in top[:5]:
            symb = str((it.get("symbol") if isinstance(it, dict) else "") or "").strip()
            w = it.get("weight") if isinstance(it, dict) else None
            try:
                if w is not None:
                    pct = float(w) * 100.0
                    lines.append(f"- {symb} ({pct:.2f}%)")
            except Exception:
                continue
        holdings_block = "\n".join(lines) if lines else "none"

    news_lines = []
    for it in (metrics.news_items or [])[:5]:
        title = (it.get("title") or "").strip()
        src = (it.get("source") or "").strip()
        dt = (it.get("publishedAt") or "")[:10]
        if title:
            news_lines.append(f"- {title} [source={src}; date={dt}]")
    news_block = "\n".join(news_lines) if news_lines else "none"

    nxt = "N/A" if inst == "etf" else (metrics.next_earnings or "unknown")

    # Schema selection based on instrument type
    if inst == "etf":
        schema = (
            '{\n'
            '  "analystRating": {\n'
            '    "rating": "string (e.g., \"Strong Buy\", \"Buy\", \"Hold\", \"Reduce\", \"Sell\")",\n'
            '    "score": "number (0-100)",\n'
            '    "trend": "string (e.g., \"Improving\", \"Stable\", \"Waning\")",\n'
            '    "rationale": "string (A concise, one-sentence explanation for the rating and trend.)"\n'
            '  },\n'
            '  "investorTakeaway": "string (The single most important, bottom-line insight for an investor. This should be the \'so what?\' of the entire analysis, delivered in a powerful, memorable sentence.)",\n'
            '  "plainEnglishSummary": "string",\n'
            '  "keyDrivers": {\n'
            '    "strength": "string",\n'
            '    "concern": "string",\n'
            '    "opportunity": "string"\n'
            '  },\n'
            '  "etfProfile": {\n'
            '    "summary": "string (Describe the ETF\'s investment strategy, e.g., \"Tracks the S&P 500 index, offering broad exposure to large-cap U.S. equities.\")",\n'
            '    "assetClass": "string (e.g., \"U.S. Equities\")",\n'
            '    "expenseRatio": "string (Provide as a percentage if available, otherwise \"N/A\")"\n'
            '  },\n'
            '  "technicalAnalysis": {\n'
            '    "pattern": "string (e.g., \'Trend within long-term channel\')",\n'
            '    "supportLevel": "string",\n'
            '    "resistanceLevel": "string",\n'
            '    "momentum": "string (e.g., \'RSI (14-day): 55 (Neutral to Positive)\')"\n'
            '  },\n'
            '  "sentimentAndNews": {\n'
            '    "newsSentiment": "string (e.g., \'Neutral\')",\n'
            '    "socialPulse": "string (e.g., \'Moderate Activity\')",\n'
            '    "analystConsensus": "string (e.g., \'Hold\')"\n'
            '  },\n'
            '  "futureProjections": {\n'
            '    "priceProjection12Mo": "string (e.g., \'$420 - $480\')",\n'
            '    "upcomingCatalysts": "array of strings (macro/sector events; e.g., [\'Fed rate decisions\', \'Sector rotation\'])"\n'
            '  }\n'
            '}'
        )
        guidance_extra = (
            "This instrument is an ETF: focus on index exposure, sector/asset-class composition, diversification, liquidity, and expense ratio if known. "
            "Emphasize macro and sector drivers rather than single-company fundamentals."
        )
    else:
        schema = (
            '{\n'
            '  "analystRating": {\n'
            '    "rating": "string (e.g., \"Strong Buy\", \"Buy\", \"Hold\", \"Reduce\", \"Sell\")",\n'
            '    "score": "number (0-100)",\n'
            '    "trend": "string (e.g., \"Improving\", \"Stable\", \"Waning\")",\n'
            '    "rationale": "string (A concise, one-sentence explanation for the rating and trend.)"\n'
            '  },\n'
            '  "investorTakeaway": "string (The single most important, bottom-line insight for an investor. This should be the \'so what?\' of the entire analysis, delivered in a powerful, memorable sentence.)",\n'
            '  "plainEnglishSummary": "string",\n'
            '  "keyDrivers": {\n'
            '    "strength": "string",\n'
            '    "concern": "string",\n'
            '    "opportunity": "string"\n'
            '  },\n'
            '  "fundamentalHealthCheck": {\n'
            '    "valuation": "object with currentPE, industryPE, and a summary label (e.g., \'Premium\')",\n'
            '    "profitability": "object with grossMargin, industryMargin, and a summary label (e.g., \'Excellent\')",\n'
            '    "growth": "object with revenueGrowthYoY, epsGrowthYoY, and a summary label (e.g., \'Exceptional\')",\n'
            '    "financialStrength": "object with debtToEquity and a summary label (e.g., \'Very Low Risk\')"\n'
            '  },\n'
            '  "technicalAnalysis": {\n'
            '    "pattern": "string (e.g., \'Detected breakout from Consolidation Channel\')",\n'
            '    "supportLevel": "string",\n'
            '    "resistanceLevel": "string",\n'
            '    "momentum": "string (e.g., \'RSI (14-day): 68 (Strong Momentum)\')"\n'
            '  },\n'
            '  "sentimentAndNews": {\n'
            '    "newsSentiment": "string (e.g., \'Bullish\')",\n'
            '    "socialPulse": "string (e.g., \'Highly Active, Mostly Positive\')",\n'
            '    "analystConsensus": "string (e.g., \'Strong Buy\')"\n'
            '  },\n'
            '  "futureProjections": {\n'
            '    "priceProjection12Mo": "string (e.g., \'$210 - $280\')",\n'
            '    "upcomingCatalysts": "array of strings (e.g., [\'Q3 FY2026 Earnings Report on Oct 29, 2025\', \'GTC 2026 Conference in March 2026\'])"\n'
            '  }\n'
            '}'
        )
        guidance_extra = (
            "This instrument is a single-company stock: you may discuss valuation, profitability, and growth only if grounded in the provided fundamentals."
        )

    anti_hallucination = (
        "Your primary directive is to act as a data-driven financial analyst. You must base your entire analysis—especially the plainEnglishSummary and keyDrivers— STRICTLY on the financial data and news headlines provided below. Do NOT make any linguistic or thematic associations based on the ticker symbol itself. For example, if the ticker is 'SPY', you must not mention espionage; if it is 'AAPL', you must not mention fruit."
    )

    instruction = (
        "You are a pragmatic, data-driven investment analyst with a focus on growth-at-a-reasonable-price (GARP). Your tone is insightful, confident, and slightly skeptical. You prioritize strong fundamentals and clear catalysts over speculative hype. Your primary directive is to synthesize the provided data into a compelling, narrative-driven analysis for a retail investor.\n"
        + anti_hallucination + "\n"
        "CRITICAL OUTPUT RULES: Return ONLY a raw JSON object exactly matching the required keys.\n"
        "- Do NOT wrap in Markdown or backticks.\n"
        "- Do NOT include any preface, explanation, or notes.\n"
        "- Base all assessments on the provided numbers and headlines; if data is missing, make conservative, clearly-labeled estimates.\n"
        "- Keep all values within reasonable bounds. analystRating.score must be a number 0-100.\n"
        "- CRITICAL COMMA RULE: Every key-value pair in an object must be followed by a comma, except for the very last one. For example: \"key1\": \"value1\", \"key2\": \"value2\" is correct; missing commas is invalid JSON.\n"
        "- For analystRating, synthesize all available data—fundamentals, technicals, and sentiment—to generate a holistic rating. The trend should reflect whether the outlook is improving or deteriorating based on recent news and performance. The rationale must be a tight summary of your reasoning.\n"
        "- For investorTakeaway, distill the entire analysis into one definitive statement. This is your expert conclusion. It should be opinionated but fair, directly answering: 'Why should I, the investor, care about this stock right now?'\n"
        + guidance_extra + "\n"
    )

    holdings_section = ""
    if inst == "etf":
        holdings_section = f"Top5Holdings:\n{holdings_block}\n"

    context = (
        f"Generated: {dt_now}\n"
        f"Symbol: {symbol.upper()}\n"
        f"InstrumentType: {inst}\n"
        f"Performance: {perf}\n"
        f"Technicals: {tech}\n"
        f"Fundamentals: {fund_line}\n"
        f"{holdings_section}"
        f"NextEarnings: {nxt}\n"
        f"RecentNews:\n{news_block}\n"
    )

    schema_block = (
        "Required JSON schema (structure and keys only; values must be concrete):\n" + schema
    )

    return instruction + "\n\n" + schema_block + "\n\nDATA:\n" + context + "\nOUTPUT: JSON ONLY"
