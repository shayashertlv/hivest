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
        "earnings": fund.get("earnings"),  # {date, actualEPS, estimatedEPS, surprisePct, bottomLine}
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
        "You are a senior financial analyst. Analyze ONLY the provided DATA block about a financial instrument "
        "(stock, ETF, index, crypto, commodity) and produce a concise, decision-oriented report in JSON.\n\n"

        "=== SENTIMENT SCORES ===\n"
        "- ALL sentiment scores MUST be numeric (0–100) and ALWAYS present.\n"
        "- sentimentDial MUST have EXACTLY three keys with EXACT names:\n"
        "  1. \"newsSentiment\"\n"
        "  2. \"socialSentiment\"\n"
        "  3. \"analystSentiment\"\n"
        "- Each key contains ONLY a {\"score\": number} field. No null, no summary, no extra fields.\n"
        "- If SocialSentiment data is missing or 'none', derive it from the news sentiment score by subtracting 3–7 points.\n"
        "- If AnalystConsensus is missing, derive analystSentiment from news, technicals.\n\n"

        "=== OUTPUT RULES ===\n"
        "1) Return exactly ONE valid JSON object — no preface, suffix, or extra text.\n"
        "2) Use English only.\n"
        "3) Use ONLY the provided DATA. If information is missing, OMIT the field entirely — EXCEPT sentiment scores, which must always be present.\n"
        "4) Convert metrics to plain language insights (e.g., 'short-term looks overbought' instead of 'RSI=72', or 'the asset moves more aggressively than the market' instead of 'beta=1.8').\n"
        "5) Ensure valid JSON syntax: double quotes only, no comments, no trailing commas.\n"
        "6) Do not assume sector, geography, or business details unless they explicitly appear in DATA.\n\n"

        "=== DATA SCOPE & HALLUCINATION RULES ===\n"
        "- Never mention external sources, banks, unless they appear in News.\n"
        "- Never invent or estimate values. If missing, omit.\n"
        "- If 'Comparables:' is absent, omit aiCompetitorAnalysis.\n\n"

        "=== CONFLICT RESOLUTION ===\n"
        "1) Prefer values with explicit timestamps nearest the DATA 'AsOf' date.\n"
        "2) Prefer structured Fundamentals JSON over narrative.\n"
        "3) Prefer consensus blocks over single mentions when tied.\n"
        "4) If conflict persists, omit the field.\n\n"

        "=== EARNINGS EXTRACTION ===\n"
        "- If Fundamentals contains 'earnings':\n"
        "  * quarterlyReport.bottomLine = earnings.bottomLine + ' +' or ' -' based on if the result is positive or negative.\n"
        "  * quarterlyReport.keyPoints is an array of strings:\n"
        "    1. \"Reported EPS of $<actual> vs. estimate of $<estimate>\"\n"
        "    2. (optional) \"Next earnings on <YYYY-MM-DD>\" — extract date from UpcomingEvents section if an 'earnings' type event exists\n"
        "  * Ensure correct JSON syntax: close quotes, arrays, braces properly.\n"
        "- If earnings are missing entirely, omit quarterlyReport.\n\n"

        "=== SENTIMENT SCORING DETAILS ===\n"
        "- News Sentiment: derive score 0–100 from tone/content. Bands:\n"
        "  75–90 = bullish, 55–74 = moderately positive, 45–54 = neutral, 25–44 = moderately negative, 10–24 = bearish.\n"
        "  If no news exists, infer a baseline 48–52 from price action/technicals.\n"
        "- Social Sentiment: mandatory numeric. Derive from news when unavailable by subtracting 3–7 points.\n"
        "- Analyst Sentiment: mandatory numeric. Use AnalystConsensus implied upside if available:\n"
        "  >15% = 70–85; 5–15% = 55–69; -5–+5% = 45–54; -15––5% = 30–44; <-15% = 15–29.\n"
        "  If missing, derive from other context or baseline 48–52.\n\n"

        "=== RISKS & OPPORTUNITIES ===\n"
        "- Provide 2–3 balanced plain-language bullets covering BOTH risks and opportunities.\n"
        "- Use NO metric terminology (e.g., beta, volatility, sharpe, etc.).\n"
        "- If nothing meaningful can be derived, omit the field entirely.\n\n"

        "=== FORMATTING RULES ===\n"
        "- Dates: use YYYY-MM-DD exactly as in DATA.\n"
        "- Currency: mirror units from DATA without conversion.\n"
        "- EPS rounding: two decimals.\n"
        "- Tone: concise, decision-oriented, no hype.\n\n"

        "=== JSON STRUCTURE ===\n"
        "{\n"
        "  \"aiDailyBrief\": \"<1–3 sentence plain-language snapshot based strictly on price/action>\",\n"
        "  \"sentimentDial\": {\n"
        "    \"newsSentiment\": { \"score\": <0–100> },\n"
        "    \"socialSentiment\": { \"score\": <0–100> },\n"
        "    \"analystSentiment\": { \"score\": <0–100> }\n"
        "  },\n"
        "  \"aiSummary\": {\n"
        "    \"quarterlyReport\": {\n"
        "      \"bottomLine\": \"<beat/miss/in line + sign>\",\n"
        "      \"keyPoints\": [ \"Reported EPS of $X.XX vs. estimate of $Y.YY\", \"Next earnings on YYYY-MM-DD\" ]\n"
        "    },\n"
        "    \"newsSummary\": \"<short bullet list of MATERIAL news only>, add a '+'/'-' if the bullet point is positive for the asset or negative for the asset respectfully \",\n"
        "    \"risksAndOpportunities\": \"<2–3 balanced bullets in plain language;>\"\n"
        "  }\n"
        "}\n\n"

        "=== FINAL VALIDATION CHECKLIST ===\n"
        "- risksAndOpportunities: plain language only, no metrics, 2–3 bullets, covers both risks and opportunities.\n"
        "- No numeric values or percentages in risksAndOpportunities.\n"
        "- Proper bracket/brace matching throughout.\n"
        "- quarterlyReport.keyPoints: each string closed, array closed with ], object closed with }, comma follows if needed.\n"
        "- All keys/strings use double quotes.\n"
        "- No truncated strings.\n"
        "- sentimentDial has EXACTLY three keys with full names and numeric scores.\n"
        "- No shortened sentiment keys, no null values, no summary fields.\n"
        "- Social sentiment derived properly if missing.\n"
        "- Output is a single valid JSON object with no extra fields.\n"
        "- Fields with 'unavailable' or 'not applicable' are omitted (except sentiment scores).\n"
        "- Earnings parsed if present, omitted if not.\n"
        "- All three sentiment scores are numeric 0–100.\n"
        "- No invented facts or metrics.\n"
        "- JSON syntax is valid.\n"
    )

    payload = (
        f"Symbol: {symbol}\n"
        f"InstrumentType: {inst_type}\n"
        f"Timeframe: {timeframe}\n"
        f"Generated: {dt_now}\n"
        f"Performance: {perf_line}\n"
        f"Benchmarks:\n{bench_block}\n"
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
