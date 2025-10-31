"""hivest/stock_analysis/llm/prompts.py"""
from __future__ import annotations
from datetime import datetime
from typing import Dict, Any, Optional, List
import json

from ..processing.models import StockAnalysisContext, StockInput, ComputedMetrics

FMT_PCT = lambda x: f"{x*100:.2f}%" if isinstance(x, (int, float)) else "n/a"
FMT_SIG = lambda x: f"{float(x):.4f}" if isinstance(x, (int, float)) else "n/a"


def _news_block(items: Optional[List[Dict[str, Any]]]) -> str:
    items = items or []
    return "\n".join(
        f"- {it.get('symbol','')}: {it.get('title','')} "
        f"[source={it.get('source','')}; date={(it.get('publishedAt','') or '')[:10]}]\n"
        f"Content: {it.get('content','')}"
        for it in items
    ) or "none"


def _fmt_value(x):
    """Make values compact and JSON-friendly for the DATA section."""
    try:
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            return x.strip()[:200]  # avoid giant blobs
        if isinstance(x, dict):
            # Preserve dicts (like earnings) but be more careful with filtering
            # Keep keys even if value is None for critical fields like bottomLine
            result = {}
            for k, v in x.items():
                if v is None:
                    # Keep None values for important fields in earnings
                    if k in ('bottomLine', 'actualEPS', 'estimatedEPS', 'date', 'surprisePct'):
                        result[k] = None
                    # Otherwise skip None values
                elif isinstance(v, (int, float)):
                    result[k] = float(v)
                elif isinstance(v, str):
                    result[k] = v.strip()
                else:
                    result[k] = v
            return result if result else {}
        return x
    except Exception:
        return x


def _fundamentals_block(fund: dict) -> str:
    """Emit a compact JSON subset of fundamentals that the LLM can reason about."""
    if not isinstance(fund, dict) or not fund:
        return "none"
    keep = {
        "price": fund.get("price"),
        "market_cap": fund.get("market_cap"),
        "pe_ttm": fund.get("pe_ttm"),
        "epsGrowthYoY": fund.get("epsGrowthYoY"),
        "grossMargin": fund.get("grossMargin"),
        "debtToEquity": fund.get("debtToEquity"),
        "analyst_target_price": fund.get("analyst_target_price"),
        "earnings": fund.get("earnings"),  # {date, actualEPS, estimatedEPS, surprisePct, bottomLine, actualRevenue, estimatedRevenue, revenueGrowthYoY, grossMargin, grossMarginYoY, operatingMargin, operatingMarginYoY, buybackAmount, nextQuarterEstimatedEPS, nextQuarterEstimatedRevenue, nextQuarterEstimateDate}
        # add more keys here if needed, but keep it compact
    }
    # Format values and filter, but be careful not to filter out earnings dict even if some fields are None
    formatted_keep = {}
    for k, v in keep.items():
        formatted_v = _fmt_value(v)
        # Don't filter out earnings dict even if it becomes empty after formatting
        if k == "earnings":
            if v is not None and v not in ("", []):
                formatted_keep[k] = formatted_v
        elif formatted_v not in (None, "", [], {}, 0):
            formatted_keep[k] = formatted_v

    result = json.dumps(formatted_keep, ensure_ascii=False) if formatted_keep else "none"

    # Debug logging to verify earnings data is included
    if fund.get("earnings"):
        print(f"[prompts] _fundamentals_block: earnings data present in fund: {fund.get('earnings')}")
        if "earnings" in formatted_keep:
            print(f"[prompts] _fundamentals_block: earnings data included in output: {formatted_keep['earnings']}")
        else:
            print(f"[prompts] _fundamentals_block: WARNING - earnings data was filtered out!")
    else:
        print(f"[prompts] _fundamentals_block: No earnings data in fund")

    return result


def _fmt_social_sentiment(data: Dict[str, Any] | None) -> str:
    """Format social sentiment snapshot from FMP into a short descriptor.
    Returns 'none' if no data available (so LLM can derive from news instead).
    """
    if not data:
        return "none"

    # FMP may return a dictionary with multiple numeric + textual fields.
    # We pick the most decision-useful ones while gracefully skipping missing values.
    parts: List[str] = []
    # Score style fields
    for key in ("score", "sentimentScore", "socialScore", "stocktwitsSentiment", "twitterSentiment"):
        val = data.get(key)
        if isinstance(val, (int, float)):
            parts.append(f"{key}={val:.2f}")
    # Volume / activity style fields
    for key in ("totalMentions", "positiveMentions", "negativeMentions", "mentions", "stocktwitsImpressions"):
        val = data.get(key)
        if isinstance(val, (int, float)) and val:
            parts.append(f"{key}={int(val)}")
    # Change/Trend descriptors
    for key in ("sentiment", "trend", "trendScore"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            parts.append(f"{key}={val.strip()}")

    if not parts:
        # As a fallback, dump the available keys with their stringified values.
        parts = [f"{k}={v}" for k, v in data.items() if v not in (None, "")]

    date_val = data.get("date") or data.get("updatedAt")
    if date_val:
        parts.append(f"date={str(date_val)[:10]}")

    return ", ".join(parts) if parts else "none"


def build_stock_prompt(ctx: Dict[str, Any]) -> str:
    """Builds an instruction + data payload for single-instrument analysis.
    Output must be a strict JSON object in English only.
    """
    si: StockInput = ctx.get("stock_input")
    cm: ComputedMetrics = ctx.get("computed_metrics")
    news_items = ctx.get("news_items") or []
    events = ctx.get("upcoming_events") or []

    symbol = getattr(si, "symbol", "").upper()
    timeframe = getattr(si, "timeframe_label", "1y")
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Benchmarks (avoid fake zeros when missing)
    bench_lines = []
    for name, obj in (getattr(cm, "benchmarks", {}) or {}).items():
        b_cum = obj.get("cum_return", None)
        rel = obj.get("relative_vs_asset", None)
        bench_lines.append(f"- {name}: cum={FMT_PCT(b_cum)}, relative_vs_asset={FMT_PCT(rel)}")
    bench_block = "\n".join(bench_lines) or "n/a"

    # Technicals (NO 0.0 fallbacks that create fake 0.00%)
    tech_line = (
        f"rsi={FMT_SIG(getattr(cm, 'rsi', None))}; "
        f"sma20={FMT_SIG(getattr(cm, 'sma20', None))}; "
        f"sma50={FMT_SIG(getattr(cm, 'sma50', None))}; "
        f"sma200={FMT_SIG(getattr(cm, 'sma200', None))}; "
        f"pct_from_52w_high={FMT_PCT(getattr(cm, 'pct_from_52w_high', None))}; "
        f"pct_from_52w_low={FMT_PCT(getattr(cm, 'pct_from_52w_low', None))}"
    )

    perf_line = (
        f"cumulative_return={FMT_PCT(getattr(cm,'cum_return',None))}; "
        f"volatility={FMT_SIG(getattr(cm,'volatility',None))}; "
        f"sharpe={FMT_SIG(getattr(cm,'sharpe',None))}; "
        f"sortino={FMT_SIG(getattr(cm,'sortino',None))}; "
        f"max_drawdown={FMT_PCT(getattr(cm,'max_drawdown',None))}; "
        f"beta={FMT_SIG(getattr(cm,'beta',None))}"
    )

    # Derived chart hints to help pattern recognition without raw charts
    try:
        last_px = getattr(cm, 'last_price', None)
        sma50 = getattr(cm, 'sma50', None)
        sma200 = getattr(cm, 'sma200', None)
        rsi_v = getattr(cm, 'rsi', None)
        p_hi = getattr(cm, 'pct_from_52w_high', None)
        p_lo = getattr(cm, 'pct_from_52w_low', None)
        trend = None
        if isinstance(sma50, (int, float)) and isinstance(sma200, (int, float)):
            if sma50 > sma200 * 1.01:
                trend = 'up'
            elif sma50 < sma200 * 0.99:
                trend = 'down'
            else:
                trend = 'sideways'
        rsi_state = None
        if isinstance(rsi_v, (int, float)):
            if rsi_v >= 70:
                rsi_state = 'overbought'
            elif rsi_v <= 30:
                rsi_state = 'oversold'
            else:
                rsi_state = 'neutral'
        hints = {
            'trend': trend,
            'above_sma50': (isinstance(last_px, (int, float)) and isinstance(sma50, (int, float)) and last_px > sma50) or False,
            'above_sma200': (isinstance(last_px, (int, float)) and isinstance(sma200, (int, float)) and last_px > sma200) or False,
            'golden_cross': (isinstance(sma50, (int, float)) and isinstance(sma200, (int, float)) and sma50 > sma200) or False,
            'near_52w_high': (isinstance(p_hi, (int, float)) and p_hi is not None and p_hi >= -0.05) or False,
            'near_52w_low': (isinstance(p_lo, (int, float)) and p_lo is not None and p_lo <= 0.05) or False,
            'rsi_state': rsi_state,
        }
        chart_hints_block = json.dumps(hints, ensure_ascii=False)
    except Exception:
        chart_hints_block = 'n/a'

    fund = getattr(cm, "fundamentals", {}) or {}
    inst_type = getattr(cm, "instrument_type", "stock")
    etf_prof = getattr(cm, "etf_profile", {}) or {}

    # Optional ETF profile block
    etf_block = "n/a"
    try:
        if str(inst_type).lower() == "etf" and isinstance(etf_prof, dict) and etf_prof:
            keep = {}
            if etf_prof.get("expense_ratio") is not None:
                try:
                    keep["expense_ratio"] = float(etf_prof.get("expense_ratio"))
                except Exception:
                    pass
            th = etf_prof.get("top_holdings") or []
            if isinstance(th, list) and th:
                # Keep top-5 holdings with symbol and weight if present
                keep["top_holdings"] = [
                    {"symbol": (h.get("symbol") or "").upper(), "weight": h.get("weight")}
                    for h in th[:5] if isinstance(h, dict)
                ]
            etf_block = json.dumps(keep, ensure_ascii=False) if keep else "present"
    except Exception:
        etf_block = "n/a"

    # Optional analyst consensus block if we have a target and last price
    target = fund.get("analyst_target_price")
    last_px = getattr(cm, "last_price", None)
    analyst_block = "none"
    if isinstance(target, (int, float)) and isinstance(last_px, (int, float)) and last_px > 0:
        implied = (target / last_px) - 1.0
        analyst_block = json.dumps({
            "target_avg": float(target),
            "last_price": float(last_px),
            "implied_upside": float(implied)  # fraction, not percent
        })

    events_block = "\n".join(
        f"- {ev.get('type','event')} on {ev.get('date','')} {ev.get('note','')}".strip() for ev in events
    ) or "none"

    instruction = (
        "You are a senior financial analyst. Work ONLY with the content inside the DATA block (a single financial instrument: stock, ETF, index, crypto, or commodity). Produce a single, concise, decision-oriented JSON report. Absolutely no text before or after the JSON.\n\n"
        "=== HARD CONSTRAINTS ===\n"
        "- Use ONLY facts present in DATA. If a field’s information is missing, OMIT that field entirely (except sentimentDial and aiSummary.newsSummary, which are always required).\n"
        "- Do NOT invent values, entities, sectors, geographies, or events.\n"
        "- English only.\n"
        "- Output EXACTLY one valid JSON object. Double quotes only, no comments, no trailing commas, no placeholders.\n"
        "- Output keys in the exact order as shown in the JSON STRUCTURE below.\n\n"
        "=== OUTPUT BOUNDARY & FORMAT (STRICT) ===\n"
        "- Your output MUST begin with '{' and end with '}'.\n"
        "- Do NOT include code fences, markdown, or any extra text before or after the JSON.\n"
        "- Do NOT emit standalone tokens/lines (like 0, 0.0, or notes) outside the JSON.\n"
        "- If a value is unknown, omit the field instead of writing placeholders.\n\n"
        "=== SENTIMENT SCORING (0–100, integers only) ===\n"
        "- sentimentDial MUST exist and MUST contain EXACTLY these keys (exact spelling):\n"
        "  1) \"newsSentiment\"\n"
        "  2) \"socialSentiment\"\n"
        "  3) \"analystSentiment\"\n"
        "- Each key is an object with ONLY {\"score\": <int 0–100>}. No floats; use whole numbers only.\n"
        "- News Sentiment:\n"
        "  • If DATA has news tone/content: map to bands → 75–90 bullish; 55–74 moderately positive; 45–54 neutral; 25–44 moderately negative; 10–24 bearish.\n"
        "  • If no news: baseline 50 derived from technicals/price if present; otherwise fixed 50.\n"
        "  • Round to nearest integer; clamp 0–100.\n"
        "- Social Sentiment:\n"
        "  • If explicit social data exists: score it 0–100 as above.\n"
        "  • If missing or \"none\": derive = newsSentiment.score − 5, then clamp 0–100.\n"
        "- Analyst Sentiment:\n"
        "  • If AnalystConsensus with implied upside exists: use mapping → >15% → 70–85; 5–15% → 55–69; −5–+5% → 45–54; −15% to −5% → 30–44; <−15% → 15–29.\n"
        "  • If missing, derive deterministically from price/technicals if present:\n"
        "      - Clear uptrend/positive momentum → 60\n"
        "      - Sideways/mixed → 50\n"
        "      - Deteriorating/downtrend → 40\n"
        "    If none are inferable, use 50. Clamp 0–100.\n\n"
        "=== CONVERSION TO PLAIN LANGUAGE ===\n"
        "- Convert raw metrics into plain insights (e.g., \"short-term looks overbought\" instead of \"RSI=72\"; \"moves more aggressively than the market\" instead of \"beta=1.8\").\n"
        "- Keep tone concise and decision-oriented; no hype.\n\n"
        "=== EARNINGS EXTRACTION (optional) ===\n"
        "Include \"quarterlyReport\" ONLY if DATA.Fundamentals.earnings exists:\n"
        "- bottomLine = one of \"beat\", \"miss\", \"in line\" based on EPS performance. Append a space and \"+\" if positive surprise; a space and \"−\" if negative surprise (e.g., \"beat +\"). For \"in line\", append nothing.\n"
        "- keyPoints is an array with the following potential elements (include only if the corresponding data is available in DATA.Fundamentals.earnings):\n"
        "  • EPS Performance (REQUIRED if actualEPS and estimatedEPS exist): \"Reported EPS of $<actual> vs. estimate of $<estimate>, beat/missed by X%\" where the percentage difference is calculated as ((actual - estimate) / |estimate|) * 100, rounded to the nearest whole number. Use \"beat by\" if actual > estimate, \"missed by\" if actual < estimate, or omit the percentage if equal.\n"
        "  • Revenue Performance (if actualRevenue exists): \"Revenue of $<X>B\" followed by:\n"
        "    - If revenueGrowthYoY exists: add \" (+X.X% YoY)\" or \" (-X.X% YoY)\"\n"
        "    - If both actualRevenue and estimatedRevenue exist: add \", beat estimates of $<Y>B by Z%\" or \", missed estimates of $<Y>B by Z%\" where Z is calculated as ((actual - estimate) / estimate) * 100, rounded to the nearest whole number. Use \"beat...by\" if actual > estimate, \"missed...by\" if actual < estimate, or omit the percentage if equal.\n"
        "    - Format revenue in billions with 1-2 decimals (e.g., \"$5.2B\"). Use \"M\" for millions if revenue < $1B.\n"
        "  • Next Quarter EPS Estimate (if nextQuarterEstimatedEPS exists): \"Next quarter estimated EPS: $<estimate>\" formatted to two decimals.\n"
        "  • Next Quarter Revenue Estimate (if nextQuarterEstimatedRevenue exists): \"Next quarter estimated revenue: $<estimate>B\" formatted in billions with 1-2 decimals (use \"M\" for millions if < $1B).\n"
        "\n"
        "IMPORTANT RULES:\n"
        "- If a data field is missing or None, OMIT that keyPoint entirely\n"
        "- Maintain the order: EPS Performance → Revenue Performance → Next Quarter EPS Estimate → Next Quarter Revenue Estimate\n"
        "- Each keyPoint must be a complete, informative sentence\n"
        "If earnings are absent, omit the entire \"quarterlyReport\" object.\n\n"
        "=== NEWS SUMMARY (MANDATORY) ===\n"
        "- \"newsSummary\" MUST ALWAYS be present as an array.\n"
        "- If material news exists in DATA, produce 1–3 short, asset-specific bullets that are actually interesting and clearly about the asset (not generic macro unless directly tied to the asset in DATA).\n"
        "- Materiality criteria: News is material if it describes substantive events, actions, or outcomes directly involving the asset, such as earnings releases, product launches, mergers/acquisitions, regulatory approvals/denials, executive changes, lawsuits, partnerships, or guidance updates. Vague mentions, passive listings in watchlists/analyses, or generic market commentary without specific asset-impacting details are NOT material.\n"
        "- Paraphrase and summarize the news events in your own words as bullet points; do not copy-paste headlines or text directly. Provide brief context or implications for the asset where relevant from DATA, keeping it concise.\n"
        "- Ensure bullets cover distinct aspects; merge closely related events into a single bullet to avoid repetition.\n"
        "- Each string must end with \"(+)\" if the bullet point is positive for the asset, \"(−)\" always end with either!\n"
        "- Use tight, informative phrasing with a clear action/outcome (\"wins contract…\", \"guidance cut…\", \"launches product…\", \"regulatory inquiry…\"). No sources/citations, no duplication, no fluff.\n"
        "- Avoid passive listings like 'featured in analysis' or 'mentioned in watchlist'; focus strictly on substantive events, actions, or outcomes directly involving the asset as described in DATA.\n"
        "- If DATA contains no news or only non-material/vague mentions, output an empty array [].\n\n"
        "=== FORMATTING RULES ===\n"
        "- Dates: YYYY-MM-DD exactly as in DATA.\n"
        "- Currency units: mirror DATA; no conversion.\n"
        "- EPS rounding: two decimals.\n"
        "- Omit any value that is missing/unknown rather than writing \"unavailable\", \"N/A\", or null.\n"
        "- Determinism: required derivations use fixed rules above (baseline 50; social = news − 5; clamps applied).\n\n"
        "=== VALIDATION CHECKLIST BEFORE YOU OUTPUT ===\n"
        "- Output is EXACTLY one JSON object; valid syntax; double quotes; no trailing commas; no extra top-level keys beyond those in the structure below.\n"
        "- \"sentimentDial\" MUST be present with EXACTLY these three keys (exact spelling, case-sensitive): \"newsSentiment\", \"socialSentiment\", \"analystSentiment\". Each must have structure {\"score\": integer}. Scores MUST be integers in [0,100], no decimals, no null values.\n"
        "- \"aiSummary.newsSummary\" present (array). If news exists in DATA → 1–3 asset-specific bullets with clear actions/outcomes; each ends with \"+\" or \"−\" for positive/negative, and no symbol for neutral. No sources, no fluff/duplicates, no passive listings.\n"
        "- If \"aiSummary.quarterlyReport\" is present: bottomLine is \"beat\"/\"miss\"/\"inline\"; EPS formatted to two decimals.\n"
        "- All three sentiment scores MUST be integers in [0,100]; Social derived = News − 5 (clamped) when social data is missing. Never output null for any sentiment score.\n"
        "- No invented facts; all content traceable to DATA.\n\n"
        "=== JSON STRUCTURE (return this shape; omit any key whose content would be empty, EXCEPT \"aiSummary.newsSummary\" which must exist and may be empty) ===\n"
        "{\n"
        "  \"sentimentDial\": {\n"
        "    \"newsSentiment\": { \"score\": 0 },\n"
        "    \"socialSentiment\": { \"score\": 0 },\n"
        "    \"analystSentiment\": { \"score\": 0 }\n"
        "  },\n"
        "  \"aiSummary\": {\n"
        "    \"newsSummary\": [],\n"
        "    \"quarterlyReport\": {\n"
        "      \"bottomLine\": \"\",\n"
        "      \"keyPoints\": []\n"
        "    }\n"
        "  }\n"
        "}\n"
    )

    # Last trading day price change
    last_px = getattr(cm, 'last_price', None)
    prev_px = getattr(cm, 'prev_close', None)
    daily_chg = getattr(cm, 'daily_change', None)
    daily_chg_pct = getattr(cm, 'daily_change_pct', None)

    price_change_line = "n/a"
    if last_px is not None and prev_px is not None and daily_chg is not None and daily_chg_pct is not None:
        price_change_line = f"last_close={last_px:.2f}, prev_close={prev_px:.2f}, change={daily_chg:+.2f} ({daily_chg_pct:+.2f}%)"
    elif last_px is not None:
        price_change_line = f"last_close={last_px:.2f} (previous close unavailable)"

    # SPY YTD benchmark
    spy_ytd = getattr(cm, 'spy_ytd_return', None)
    spy_ytd_line = FMT_PCT(spy_ytd) if spy_ytd is not None else "n/a"

    payload = (
        f"Symbol: {symbol}\n"
        f"InstrumentType: {inst_type}\n"
        f"Timeframe: {timeframe}\n"
        f"Generated: {dt_now}\n"
        f"LastTradingDayPriceChange: {price_change_line}\n"
        f"Performance: {perf_line}\n"
        f"Benchmarks:\n{bench_block}\n"
        f"SPY_YTD_Performance: {spy_ytd_line}\n"
        f"Technicals: {tech_line}\n"
        f"ChartHints: {chart_hints_block}\n"
        f"Fundamentals: {_fundamentals_block(fund)}\n"
        f"SocialSentiment: {_fmt_social_sentiment(fund.get('social_sentiment'))}\n"
        f"AnalystConsensus: {analyst_block}\n"
        f"ETFProfile: {etf_block}\n"
        f"UpcomingEvents:\n{events_block}\n"
        f"News (titles and content):\n{_news_block(news_items)}\n"
    )
    return instruction + "\n\n--- DATA ---\n" + payload
