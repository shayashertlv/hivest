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

        "**ABSOLUTE REQUIREMENT: ALL SENTIMENT SCORES MUST BE NUMERIC 0-100**\n"
        "BEFORE YOU OUTPUT ANYTHING: Verify that sentimentDial has EXACTLY THREE keys with EXACT names:\n"
        "  1. \"newsSentiment\" (NOT \"news\") - with numeric score 0-100\n"
        "  2. \"socialSentiment\" (NOT \"social\") - with numeric score 0-100\n"
        "  3. \"analystSentiment\" (NOT \"analyst\") - with numeric score 0-100\n"
        "If you see 'SocialSentiment: none' in the DATA, you MUST derive socialSentiment.score from newsSentiment.score minus 3-7 points.\n"
        "NEVER use shortened key names. NEVER output null. NEVER add 'summary' fields. Only 'score' with a number 0-100.\n\n"

        "**CORE OUTPUT RULES**\n"
        "1) JSON ONLY: Return exactly one valid JSON object. No preface/suffix text.\n"
        "2) ENGLISH ONLY: All strings must be English.\n"
        "3) USE ONLY PROVIDED DATA: Never infer or fetch facts. If a field has no information available, OMIT IT ENTIRELY from the JSON output. "
        "   EXCEPTION: sentiment scores in sentimentDial MUST ALWAYS be numeric 0-100 and MUST ALWAYS be present, "
        "   derive them from available data (news, price action, technicals) even if direct data is missing.\n"
        "4) PLAIN-LANGUAGE INSIGHTS: Convert metrics into conclusions (e.g., say \"short-term looks overbought\" rather than \"RSI=72\"). "
        "   Do not include raw metric names/values unless the schema explicitly requires it.\n"
        "5) VALID JSON GUARANTEE: Double quotes only, no comments, no trailing commas. "
        "   Every '{' has exactly one matching '}'.\n"
        "6) ASSET-AGNOSTIC: Do not assume sector, country, or business details unless explicitly included in DATA.\n\n"

        "**STRICT ANTI-HALLUCINATION & DATA SCOPE**\n"
        "- Never mention sources, banks, analysts, or quotes unless they appear in the provided News list.\n"
        "- Never claim index comparisons (e.g., \"outperformed S&P 500\") unless explicitly stated in DATA.\n"
        "- If a metric (beta, EPS, price, targets) is absent, do not estimate—OMIT THE FIELD ENTIRELY from the output.\n"
        "- If the DATA lacks 'Comparables:', omit \"aiCompetitorAnalysis\" from the output.\n"
        "- EXCEPTION FOR SENTIMENT SCORES: sentimentDial scores are NEVER null and NEVER omitted. If DATA lacks 'AnalystConsensus:', derive analystSentiment "
        "  from news/technicals. If DATA lacks 'SocialSentiment:', derive socialSentiment from news score minus 3-7 points.\n\n"

        "**CONFLICT-HANDLING & TRUST ORDER**\n"
        "When multiple values for the same metric appear:\n"
        "1) Prefer items with explicit timestamps closest to the 'AsOf' date in DATA.\n"
        "2) Prefer structured 'Fundamentals:' JSON over narrative text in news.\n"
        "3) If timestamps tie, prefer consensus/aggregated blocks over single mentions.\n"
        "4) If conflicts persist, OMIT THE FIELD ENTIRELY from the output (do not invent a value).\n\n"

        "**EARNINGS EXTRACTION (CRITICAL)**\n"
        "- The 'Fundamentals:' line includes a JSON object. If it has 'earnings', you MUST populate quarterlyReport from it:\n"
        "  * quarterlyReport.bottomLine = earnings.bottomLine ('beat'/'miss'/'in line')\n"
        "  * quarterlyReport.keyPoints is an ARRAY. MANDATORY SYNTAX RULES:\n"
        "    1. Each item must be a complete string enclosed in double quotes\n"
        "    2. Items are separated by commas\n"
        "    3. The array MUST close with ] immediately after the last quoted string\n"
        "    4. After the ], you MUST add a comma and close the quarterlyReport object with }\n"
        "  * Content for keyPoints:\n"
        "    - First item: \"Reported EPS of $<actualEPS> vs. estimate of $<estimatedEPS>\"\n"
        "    - Second item (if date exists): \"Results released on <YYYY-MM-DD>\"\n"
        "  * CRITICAL: Verify each keyPoint string is COMPLETE before adding comma and closing bracket.\n"
        "- Only if 'earnings' is entirely missing in Fundamentals JSON: omit quarterlyReport entirely from the JSON output.\n"
        "- Do NOT write 'unavailable' if the earnings block exists—parse it.\n"
        "- Example valid structure:\n"
        "  \"quarterlyReport\": {\n"
        "    \"bottomLine\": \"beat\",\n"
        "    \"keyPoints\": [\n"
        "      \"Reported EPS of $1.68 vs. estimate of $1.31\",\n"
        "      \"Results released on 2025-07-31\"\n"
        "    ]\n"
        "  },\n"
        "  \"newsSummary\": \"...\"\n\n"

        "**SENTIMENT SCORING RULES (ALL SCORES MANDATORY 0-100)**\n"
        "CRITICAL: ALL three sentiment scores MUST ALWAYS be numeric values 0–100. NEVER EVER use null for any sentiment score.\n"
        "CRITICAL: sentimentDial ONLY contains \"score\" fields with numeric values. NEVER add \"summary\" or any other fields.\n\n"
        "- News Sentiment (ALWAYS REQUIRED): Score 0–100 based on headline tone + content across articles. "
        "  Use these bands: 75–90 predominantly positive/bullish; 55–74 moderately positive; 45–54 mixed/neutral; "
        "  25–44 moderately negative; 10–24 predominantly negative/bearish. "
        "  If no news is provided, infer a baseline score (48-52) from price performance, technical indicators, and available data. "
        "  Never mention analysts under 'newsSentiment'. Output ONLY: {\"score\": <number>}\n"
        "- Social Sentiment (ABSOLUTELY MANDATORY NUMERIC SCORE): When SocialSentiment data is 'none' or missing, you MUST derive "
        "  a score from the news sentiment. If news score is 65, set social to 60-63 (slightly more conservative). If news is 70, set social to 65-68. "
        "  If news is 50, set social to 48-52. ALWAYS mirror the news sentiment directionally but be 3-7 points more conservative. "
        "  This is NOT optional—you MUST provide a number even if social data is unavailable. "
        "  NEVER output score=null or add a 'summary' field. Output ONLY: {\"score\": <number>}\n"
        "- Analyst Sentiment (ALWAYS REQUIRED): You MUST always deliver a numeric score 0–100 for analystSentiment. "
        "  If AnalystConsensus with implied_upside is present: Score based on implied upside (>15% upside = 70-85; 5-15% = 55-69; "
        "  -5 to +5% = 45-54; -15 to -5% = 30-44; <-15% = 15-29). "
        "  If AnalystConsensus is missing or incomplete, derive a score from news sentiment, technical trends, and fundamentals "
        "  (use neutral 48-52 if insufficient data). NEVER output score=null. Output ONLY: {\"score\": <number>}\n\n"

        "**RISK/OPPORTUNITY INTERPRETATION**\n"
        "- CRITICAL: NEVER use metric terminology in riskOpportunityProfile fields. Forbidden terms include: 'beta', 'sharpe ratio', "
        "  'sortino', 'alpha', 'volatility', 'standard deviation', 'correlation', 'R-squared', 'drawdown', 'VaR', etc.\n"
        "- Convert ALL metrics to plain language conclusions that a non-technical reader can understand.\n"
        "- marketVolatility: infer conclusion (e.g., 'The stock shows higher sensitivity to overall market movements') from provided "
        "  volatility/beta data—NEVER print metric names or numeric values like 'beta=1.2' or 'volatility 25%'. If no data available, OMIT THIS FIELD.\n"
        "- concentrationRisk: state dependency on a single product/program only if DATA indicates it; otherwise OMIT THIS FIELD.\n"
        "- regulatoryRisk: summarize only if explicitly present in DATA; otherwise OMIT THIS FIELD.\n"
        "- growthPotential: list forward drivers/tailwinds only if they appear in DATA (pipelines, partnerships, catalysts, TAM notes); otherwise OMIT THIS FIELD.\n\n"

        "**CONSISTENCY & FORMATTING**\n"
        "- Dates: if you include a date, use YYYY-MM-DD as given in DATA (no assumptions).\n"
        "- Currency/Units: do not convert or assume units; mirror the units present in DATA text.\n"
        "- Rounding: keep two decimals for EPS when you form the key point string (e.g., $-0.19). Do not round other values unless provided.\n"
        "- Tone: concise, decision-oriented, no hype words unless they appear in the given articles.\n\n"

        "**WORKED EXAMPLE: DERIVING SOCIAL SENTIMENT FROM NEWS**\n"
        "Scenario: DATA shows 'SocialSentiment: none' but news sentiment evaluates to 78 (as in your current case).\n"
        "Step 1: Calculate social = news - (3 to 7) = 78 - 5 = 73\n"
        "Step 2: Output \"socialSentiment\": { \"score\": 73 }\n"
        "NEVER output \"socialSentiment\": { \"score\": null, \"summary\": \"unavailable\" } — this violates the requirements.\n\n"

        "**REQUIRED JSON STRUCTURE (MUST MATCH EXACTLY)**\n"
        "CRITICAL: Pay close attention to bracket matching. Every [ needs ], every { needs }, in the correct order.\n"
        "CRITICAL: sentimentDial must have EXACTLY these three keys (use full names, not shortened):\n"
        "  - \"newsSentiment\" (NOT \"news\")\n"
        "  - \"socialSentiment\" (NOT \"social\")\n"
        "  - \"analystSentiment\" (NOT \"analyst\")\n"
        "CRITICAL: Each key contains ONLY a score field with a number 0-100. NO null, NO summary, NO other fields.\n\n"
        "CORRECT EXAMPLE when SocialSentiment data is 'none':\n"
        "  \"sentimentDial\": {\n"
        "    \"newsSentiment\": { \"score\": 78 },\n"
        "    \"socialSentiment\": { \"score\": 73 },\n"
        "    \"analystSentiment\": { \"score\": 50 }\n"
        "  }\n\n"
        "INCORRECT EXAMPLES:\n"
        "  ❌ \"sentimentDial\": { \"news\": { \"score\": 78 }, ... }  // Wrong key name\n"
        "  ❌ \"socialSentiment\": { \"score\": null, \"summary\": \"unavailable\" }  // Has null and summary\n"
        "  ❌ \"socialSentiment\": { \"score\": null }  // Has null\n\n"
        "{\n"
        "  \"aiDailyBrief\": \"<1–3 sentence snapshot for today/yesterday, derived strictly from provided price/action context>\",\n"
        "  \"sentimentDial\": {\n"
        "    \"newsSentiment\":    { \"score\": <0-100 REQUIRED—NEVER null> },\n"
        "    \"socialSentiment\":  { \"score\": <0-100 REQUIRED—NEVER null, derive from news if no social data> },\n"
        "    \"analystSentiment\": { \"score\": <0-100 REQUIRED—NEVER null> }\n"
        "  },\n"
        "  \"aiSummary\": {\n"
        "    \"quarterlyReport\": {\n"
        "      \"bottomLine\": \"<'beat'/'miss'/'in line' from Fundamentals.earnings.bottomLine; omit quarterlyReport entirely if no earnings data>\",\n"
        "      \"keyPoints\": [\n"
        "        \"Reported EPS of $X.XX vs. estimate of $Y.YY\",\n"
        "        \"Results released on YYYY-MM-DD\"\n"
        "      ]\n"
        "    },\n"
        "    \"newsSummary\": \"<3–4 line synthesis of major events/themes; acknowledge consensus vs. disagreements where relevant>\"\n"
        "  },\n"
        "  \"riskOpportunityProfile\": {\n"
        "    \"marketVolatility\":   \"<plain language conclusion only—NO metric names like 'beta'/'volatility', NO numbers; OMIT THIS FIELD if no data>\",\n"
        "    \"concentrationRisk\":  \"<dependency conclusion in plain language; OMIT THIS FIELD if no data>\",\n"
        "    \"regulatoryRisk\":     \"<plain language summary from DATA; OMIT THIS FIELD if no data>\",\n"
        "    \"growthPotential\":    \"<plain language drivers from DATA; OMIT THIS FIELD if no data>\"\n"
        "  }\n"
        "}\n\n"

        "FINAL VALIDATION CHECKLIST (BEFORE YOU OUTPUT):\n"
        "- [ ] CRITICAL: riskOpportunityProfile contains ZERO metric terminology (no 'beta', 'sharpe ratio', 'volatility', 'sortino', "
        "      'alpha', 'standard deviation', 'correlation', 'drawdown', etc.). All descriptions are in plain non-technical language.\n"
        "- [ ] riskOpportunityProfile contains NO numeric values or percentages—only qualitative conclusions.\n"
        "- [ ] BRACKET MATCHING: Every opening [ has a closing ], every opening { has a closing }, in the correct order.\n"
        "- [ ] quarterlyReport.keyPoints array:\n"
        "      * Each string item is COMPLETE and properly closed with a closing double-quote\n"
        "      * Array is properly closed with ] after the last item\n"
        "      * quarterlyReport object has closing } after the array\n"
        "      * Comma follows the } to separate from newsSummary\n"
        "- [ ] All string values are properly quoted with double quotes, including all keys.\n"
        "- [ ] NO TRUNCATED STRINGS: Every opening quote \" has a matching closing quote \" on the same logical value.\n"
        "- [ ] CRITICAL CHECK: sentimentDial has EXACTLY three keys: \"newsSentiment\", \"socialSentiment\", \"analystSentiment\" (full names, not shortened).\n"
        "- [ ] CRITICAL CHECK: NO shortened keys like \"news\", \"social\", or \"analyst\" - use full names only.\n"
        "- [ ] CRITICAL CHECK: Each sentiment contains ONLY {\"score\": <number>}. NO null, NO 'summary', NO other fields.\n"
        "- [ ] CRITICAL CHECK: If SocialSentiment data was 'none', you MUST have derived socialSentiment score from news (news_score minus 3-7 points).\n"
        "- [ ] Output is a single JSON object with the exact required keys and nesting.\n"
        "- [ ] No extra fields were added.\n"
        "- [ ] CRITICAL: Fields with 'unavailable' or 'not applicable' values are OMITTED from the JSON output (EXCEPT sentiment scores which must ALWAYS be numeric).\n"
        "- [ ] QuarterlyReport correctly parsed from Fundamentals.earnings if present; otherwise omit entirely.\n"
        "- [ ] News Sentiment: ALWAYS has a numeric score 0–100 (NEVER null). Score derived from news, or inferred from price/technicals.\n"
        "- [ ] Social Sentiment: ALWAYS has a numeric score 0–100 (NEVER null). If no social data, derive from news score minus 3-7 points.\n"
        "- [ ] Analyst Sentiment: ALWAYS has a numeric score 0–100 (NEVER null). Score derived from analyst consensus, news, or baseline inference.\n"
        "- [ ] ALL THREE sentiment scores are numeric 0-100—ZERO null values anywhere in sentimentDial.\n"
        "- [ ] No invented comparisons, targets, or metrics. All claims trace to the DATA input.\n"
        "- [ ] Validate JSON syntax: Use a mental JSON parser to verify every quote, comma, bracket, and brace is in the right place.\n"
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
        f"SocialSentiment: {_fmt_social_sentiment(fund)}\n"
        f"AnalystConsensus: {analyst_block}\n"
        f"ETFProfile: {etf_block}\n"
        f"UpcomingEvents:\n{events_block}\n"
        f"News (titles and content):\n{_news_block(news_items)}\n"
    )
    return instruction + "\n\n--- DATA ---\n" + payload
