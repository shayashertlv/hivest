import os
import sys
import time
import random
import traceback
import requests
import json
from dataclasses import asdict, is_dataclass
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv, find_dotenv

# Resolve project root relative to this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Find the nearest `.env` walking up from this file; fall back to project root
dotenv_path = find_dotenv(usecwd=False)
if not dotenv_path:
    dotenv_path = os.path.join(PROJECT_ROOT, ".env")

# Load it (override ensures PyCharm’s run config doesn't mask changes)
load_dotenv(dotenv_path=dotenv_path, override=True)

# --- Optional debug (remove after it works) ---
print("Using .env:", dotenv_path)
print("OPENROUTER_API_KEY present:", bool(os.getenv("OPENROUTER_API_KEY")))
app = Flask(__name__)

# Ensure package imports work when running this file directly: `python hivest\\api_openrouter_testing.py`
if __package__ is None or __package__ == "":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from hivest.portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from hivest.portfolio_analysis.processing.models import PortfolioInput, AnalysisOptions as PortfolioAnalysisOptions
from hivest.portfolio_analysis.llm.prompts import build_portfolio_prompt
try:
    # Optional fallback text generator for portfolios
    from hivest.portfolio_analysis.llm.prompts import _fallback_render as _portfolio_fallback_render
except Exception:
    _portfolio_fallback_render = None

# Stock Analysis
from hivest.stock_analysis.processing.engine import analyze_stock
from hivest.stock_analysis.processing.models import StockInput, AnalysisOptions as StockAnalysisOptions
from hivest.stock_analysis.llm.prompts import build_stock_prompt


# --- OpenRouter LLM helper (replicates llm_client.make_llm flow, but for OpenRouter) ---

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL="openai/gpt-oss-120b"

def _get_openrouter_model(passed: str | None = None) -> str:
    # Priority: explicit param -> env -> default mapping similar to llama3:8b
    if passed and isinstance(passed, str) and passed.strip():
        model_in = passed.strip()
    else:
        model_in = (os.getenv("OPENROUTER_MODEL", "").strip() or "")

    if not model_in:
        # Default to a close analogue of llama3:8b instruct on OpenRouter
        return "openai/gpt-oss-120b"

    # Light mapping for common shorthand used by api.py ("llama3:8b")
    low = model_in.lower()
    if low.startswith("llama3") or "llama3:8b" in low:
        return "meta-llama/llama-3.1-8b-instruct"
    return model_in


def _get_openrouter_timeout() -> int:
    val = os.getenv("OPENROUTER_TIMEOUT", "180")
    try:
        return int(val)
    except Exception:
        return 180


def _get_openrouter_temperature() -> float:
    val = os.getenv("OPENROUTER_TEMPERATURE", "0.4")
    try:
        return float(val)
    except Exception:
        return 0.4


def _get_openrouter_config() -> dict:
    """Returns a dictionary with OpenRouter timeout and temperature settings."""
    try:
        timeout = int(os.getenv("OPENROUTER_TIMEOUT", "180"))
    except (ValueError, TypeError):
        timeout = 180
    try:
        temperature = float(os.getenv("OPENROUTER_TEMPERATURE", "0.2"))
    except (ValueError, TypeError):
        temperature = 0.2
    cfg = {"timeout": timeout, "temperature": temperature}
    print(f"[llm] OpenRouter config loaded: timeout={cfg['timeout']}s, temperature={cfg['temperature']}")
    return cfg


def _is_openrouter_configured() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY", "").strip())


def _extract_json_object(text: str):
    """Robustly extract a single top-level JSON object from model output.

    Strategy:
    1) Fast path: direct json.loads
    2) Extract the first balanced {...} object (brace matching with string awareness)
    3) Attempt progressive repairs:
       - Strip code fences/backticks and BOM/zero-width spaces
       - Normalize smart quotes to standard double quotes
       - Remove trailing commas before } or ]
       - Remove stray standalone numeric lines between properties (common LLM glitch)
       - Insert missing commas between object members when a newline separates members without a comma
    4) Try up to 3 repair passes; return None if all fail.
    """
    import re

    if not isinstance(text, str):
        return None

    def _clean_base(s: str) -> str:
        # Remove code fences and non-printing characters
        s = s.replace("```json", "").replace("```", "")
        s = s.replace("\ufeff", "")  # BOM
        s = s.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
        s = s.replace("\xa0", " ")
        # Normalize curly quotes to straight quotes
        s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u201e", '"').replace("\u201f", '"')
        s = s.replace("\u2018", "'").replace("\u2019", "'")
        return s

    def _find_first_object(s: str) -> str | None:
        in_str = False
        esc = False
        depth = 0
        start = -1
        for i, ch in enumerate(s):
            if in_str:
                if esc:
                    esc = False
                elif ch == '\\':
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            else:
                if ch == '"':
                    in_str = True
                elif ch == '{':
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == '}':
                    if depth > 0:
                        depth -= 1
                        if depth == 0 and start != -1:
                            return s[start:i+1]
        return None

    def _remove_trailing_commas(s: str) -> str:
        return re.sub(r',\s*([}\]])', r'\1', s)

    def _remove_standalone_number_lines(s: str) -> str:
        # Remove lines that contain only a number (int/float) possibly followed by a comma.
        return re.sub(r'^[\t ]*[+-]?(?:\d+(?:\.\d+)?|\.\d+)[\t ]*,?[\t ]*\r?\n', '', s, flags=re.MULTILINE)

    def _insert_missing_commas_between_members(s: str) -> str:
        # If a line ends a value without a comma and the next non-space starts a quoted key, insert a comma.
        # Heuristic: look for patterns like "}\n\s*\"" or number/true/false/null closing a member.
        s = re.sub(r'(\}|\]|\d|true|false|null)\s*\n\s*(\")', r'\1,\n\2', s)
        return s

    # Try direct parse first
    txt = _clean_base(text.strip())
    try:
        return json.loads(txt)
    except Exception:
        pass

    # Extract first object region
    obj_region = _find_first_object(txt)
    if obj_region is None:
        return None

    candidates = [obj_region]

    # Progressive repairs
    for i in range(3):
        cur = candidates[-1]
        for fixer in (_remove_trailing_commas, _remove_standalone_number_lines, _insert_missing_commas_between_members):
            fixed = fixer(cur)
            try:
                return json.loads(fixed)
            except Exception:
                cur = fixed
        candidates.append(cur)

    # Final attempt without repairs using the region only
    try:
        return json.loads(obj_region)
    except Exception:
        return None


def _to_jsonable(obj):
    """Convert dataclasses and complex types to JSON-serializable structures."""
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    return obj


# --- Sanitizer helpers & constants for stock-analysis JSON ---
_def_unavailable = "unavailable"


def _clamp_score(x):
    try:
        if x is None:
            return None
        v = float(x)
        if v != v:  # NaN
            return None
        return max(0.0, min(100.0, v))
    except Exception:
        return None


def _to_float_safe(v):
    """Parse to float without applying unit conversions. Returns None on failure."""
    try:
        if isinstance(v, str):
            s = v.strip()
            if s.endswith('%'):
                # keep as string; percent handling belongs in the normalizer
                s = s[:-1].strip()
            return float(s)
        return float(v)
    except Exception:
        return None


def _score_0_to_100_from_any(x):
    """
    Normalize sentiment-like values into [0..100].
    - If a percent string (e.g., "56%"), treat as 56.
    - If hintable/domains: [-1..1] -> (x+1)*50, [0..1] -> x*100, [0..100] -> clamp.
    - Small floats like 1.2 are interpreted as 0..1 domain (guardrail) unless integer-like.
    Returns float or None.
    """
    hint = None
    if isinstance(x, str) and x.strip().endswith('%'):
        hint = 'percent'
    score, meta = _normalize_score_0_100(x, hint=hint)
    return score

def _normalize_score_0_100(x, *, hint=None):
    """Return (score_0_100|None, meta). Does not raise.
    hint in { 'percent', '0_to_1', 'minus1_to_1', '0_to_100', None }
    """
    meta = { 'hint': hint, 'source': None, 'warning': None }
    if x is None:
        return None, {**meta, 'reason': 'missing'}

    # Raw percent string detection is handled by caller; we only get numbers here.
    fx = _to_float_safe(x)
    if fx is None:
        return None, {**meta, 'reason': 'type_error'}

    # Apply hint first
    if hint == 'percent':
        # 56 means 56%, clamp 0..100
        return _clamp_score(fx), {**meta, 'source': 'hint_percent'}
    if hint == '0_to_1':
        return _clamp_score(fx * 100.0), {**meta, 'source': 'hint_0_1'}
    if hint == 'minus1_to_1':
        return _clamp_score((fx + 1.0) * 50.0), {**meta, 'source': 'hint_-1_1'}
    if hint == '0_to_100':
        return _clamp_score(fx), {**meta, 'source': 'hint_0_100'}

    # Heuristic detection when no hint
    # If it came from a string ending with '%', caller should have passed hint='percent'.
    if -1.0 <= fx <= 1.0:
        # Treat small non-integers near 1 as scores in 0..1
        if 0.0 <= fx <= 1.0:
            return _clamp_score(fx * 100.0), {**meta, 'source': 'detected_0_1'}
        return _clamp_score((fx + 1.0) * 50.0), {**meta, 'source': 'detected_-1_1'}

    if 0.0 < fx < 1.5 and not float(fx).is_integer():
        # protect 1.2 style small floats from being misread as 0..100
        return _clamp_score(min(fx, 1.0) * 100.0), {**meta, 'source': 'heuristic_0_1_small_float'}

    if 0.0 <= fx <= 100.0:
        return _clamp_score(fx), {**meta, 'source': 'detected_0_100'}

    return None, {**meta, 'reason': 'out_of_range'}

def _derive_social_sentiment_from_fmp(social_obj: dict) -> tuple:
    """
    Best-effort extraction from FMP /v4/social-sentiment payloads using a canonical 0–100 score.
    Tries (in order):
      1) bullish/bearish percent fields
      2) 'sentimentScore' (expected in [-1..1])
      3) 'score' (generic)
      4) positive/negative counts ratio
    Returns (score_0_100 or None, summary_text). In debug mode (HIVEST_DEBUG_STOCK=1), summary includes failure reasons.
    """
    if not isinstance(social_obj, dict):
        print("[stock] Social sentiment payload is not a dict; marking unavailable")
        return None, "unavailable"

    # Debug mode for verbose reasons in summary
    debug_mode = False
    try:
        debug_mode = (os.getenv('HIVEST_DEBUG_STOCK', '0').strip() == '1')
    except Exception:
        pass

    try:
        print(f"[stock] Social sentiment payload keys: {list(social_obj.keys())}")
    except Exception:
        pass

    failure_reasons = []

    # 1) bullish percentage style
    bull_key = None
    bull_score = None
    bull_meta = None
    for k in ("bullishPercent", "bullish_percentage", "bullish"):
        if k in social_obj:
            bull_key = k
            v = social_obj.get(k)
            hint = 'percent' if isinstance(v, str) and v.strip().endswith('%') else None
            bull_score, bull_meta = _normalize_score_0_100(v, hint=hint)
            break
    if bull_score is not None:
        summary = f"FMP social sentiment shows ~{bull_score:.0f}% bullish."
        print(f"[stock] Social sentiment derived from {bull_key}: score={bull_score}, source={bull_meta.get('source')}")
        return bull_score, summary
    else:
        if bull_key is not None:
            failure_reasons.append(f"{bull_key}={ (bull_meta or {}).get('reason', 'invalid') }")
        else:
            failure_reasons.append("bullishPercent=missing")

    # 2) explicit sentimentScore (often -1..1)
    sent_key = None
    for k in ("sentimentScore", "sentiment_score"):
        if k in social_obj:
            sent_key = k
            s, meta = _normalize_score_0_100(social_obj[k], hint='minus1_to_1')
            if s is not None:
                summary = ("FMP social sentiment indicates positive tilt." if s >= 60 else
                           ("neutral tilt." if s >= 40 else "negative tilt."))
                print(f"[stock] Social sentiment derived from '{k}': score={s}, source={meta.get('source')}")
                return s, summary
            else:
                failure_reasons.append(f"{k}={meta.get('reason','invalid')}")
            break
    if sent_key is None:
        failure_reasons.append("sentimentScore=missing")

    # 2b) FMP-specific sentiment fields: stocktwitsSentiment, twitterSentiment
    st_sent = social_obj.get("stocktwitsSentiment")
    tw_sent = social_obj.get("twitterSentiment")
    valid_scores = []
    if st_sent is not None:
        s, meta = _normalize_score_0_100(st_sent, hint='minus1_to_1')
        if s is not None:
            valid_scores.append(("stocktwitsSentiment", s, meta))
            print(f"[stock] Found stocktwitsSentiment: raw={st_sent}, normalized={s}, source={meta.get('source')}")
        else:
            failure_reasons.append(f"stocktwitsSentiment={meta.get('reason','invalid')}")
    if tw_sent is not None:
        s, meta = _normalize_score_0_100(tw_sent, hint='minus1_to_1')
        if s is not None:
            valid_scores.append(("twitterSentiment", s, meta))
            print(f"[stock] Found twitterSentiment: raw={tw_sent}, normalized={s}, source={meta.get('source')}")
        else:
            failure_reasons.append(f"twitterSentiment={meta.get('reason','invalid')}")

    if valid_scores:
        # Average if both present, otherwise use the single value
        avg_score = sum(score for _, score, _ in valid_scores) / len(valid_scores)
        sources = ", ".join(name for name, _, _ in valid_scores)
        summary = ("Social sentiment indicates positive tilt." if avg_score >= 60 else
                   ("neutral tilt." if avg_score >= 40 else "negative tilt."))
        print(f"[stock] Social sentiment derived from {sources}: averaged_score={avg_score:.1f}")
        return avg_score, summary

    if st_sent is None:
        failure_reasons.append("stocktwitsSentiment=missing")
    if tw_sent is None:
        failure_reasons.append("twitterSentiment=missing")

    # 3) generic 'score'
    if "score" in social_obj:
        s, meta = _normalize_score_0_100(social_obj["score"])
        if s is not None:
            print(f"[stock] Social sentiment derived from generic 'score': score={s}, source={meta.get('source')}")
            return s, "FMP social sentiment score available."
        else:
            failure_reasons.append(f"score={meta.get('reason','invalid')}")
    else:
        failure_reasons.append("score=missing")

    # 4) positive/negative counts ratio
    pos = None; neg = None; pos_k = None; neg_k = None
    for k in ("positive", "positiveMention", "pos", "twitterPositive", "redditPositive"):
        if k in social_obj:
            pos_k = k
            pos = _to_float_safe(social_obj[k])
            break
    for k in ("negative", "negativeMention", "neg", "twitterNegative", "redditNegative"):
        if k in social_obj:
            neg_k = k
            neg = _to_float_safe(social_obj[k])
            break
    if pos is not None and neg is not None and (pos + neg) > 0:
        pct = (pos / (pos + neg)) * 100.0
        score = _clamp_score(pct)
        print(f"[stock] Social sentiment derived from pos/neg ratio: score={score}, pct_pos={pct:.0f}%")
        return score, f"Approx. {pct:.0f}% positive mentions."
    else:
        if pos_k is None:
            failure_reasons.append("positive=missing")
        elif pos is None:
            failure_reasons.append(f"{pos_k}=type_error")
        if neg_k is None:
            failure_reasons.append("negative=missing")
        elif neg is None:
            failure_reasons.append(f"{neg_k}=type_error")
        if pos is not None and neg is not None and (pos + neg) == 0:
            failure_reasons.append("pos+neg=zero")

    print("[stock] Social sentiment unavailable after heuristics")
    summary = "unavailable"
    if debug_mode and failure_reasons:
        summary = f"unavailable (no valid fields: {', '.join(failure_reasons)})"
    return None, summary

def _clamp_score_ex(x):
    try:
        if x is None:
            return None, 'missing'
        v = float(x)
        if v != v:  # NaN
            return None, 'nan'
        return max(0.0, min(100.0, v)), None
    except Exception:
        return None, 'type_error'

def _sanitize_stock_output(ctx: dict, prompt: str, parsed: dict) -> dict:
    parsed = parsed or {}
    out = json.loads(json.dumps(parsed))  # deep copy

    # Get or create sentimentDial - but first clean it of any incorrect keys
    sd_raw = out.get("sentimentDial", {})

    # Debug: log what LLM returned before sanitization
    try:
        print(f"[stock] _sanitize: RAW sentimentDial from LLM: {sd_raw}")
    except Exception:
        pass

    # Extract scores from whatever structure LLM created, handling common mistakes
    ns_score = None
    ss_score = None
    an_score = None

    if isinstance(sd_raw, dict):
        # Try to extract newsSentiment score
        if "newsSentiment" in sd_raw:
            ns_obj = sd_raw["newsSentiment"]
            ns_score = ns_obj.get("score") if isinstance(ns_obj, dict) else None
        elif "newsSent" in sd_raw:  # Common LLM mistake - truncated key
            news_obj = sd_raw["newsSent"]
            ns_score = news_obj.get("score") if isinstance(news_obj, dict) else news_obj
        elif "news" in sd_raw:  # Common mistake
            news_obj = sd_raw["news"]
            ns_score = news_obj.get("score") if isinstance(news_obj, dict) else news_obj

        # Try to extract socialSentiment score
        if "socialSentiment" in sd_raw:
            ss_obj = sd_raw["socialSentiment"]
            ss_score = ss_obj.get("score") if isinstance(ss_obj, dict) else None
        elif "social" in sd_raw:  # Common mistake
            social_obj = sd_raw["social"]
            ss_score = social_obj.get("score") if isinstance(social_obj, dict) else social_obj

        # Try to extract analystSentiment score
        if "analystSentiment" in sd_raw:
            an_obj = sd_raw["analystSentiment"]
            an_score = an_obj.get("score") if isinstance(an_obj, dict) else None
        elif "analyst" in sd_raw:  # Common mistake
            analyst_obj = sd_raw["analyst"]
            an_score = analyst_obj.get("score") if isinstance(analyst_obj, dict) else analyst_obj

    print(f"[stock] _sanitize: Extracted scores - news={ns_score}, social={ss_score}, analyst={an_score}")

    # Create clean sentiment objects
    ns = {"score": ns_score}
    ss = {"score": ss_score}
    an = {"score": an_score}

    # Enforce comps/analyst nulls when prompt declares none
    if "Comparables: none" in prompt:
        out["aiCompetitorAnalysis"] = []
    # Note: We no longer null out analyst sentiment when AnalystConsensus is none
    # The LLM is now required to ALWAYS provide a numeric score by deriving from other data

    # Analyst sentiment: derive from AnalystConsensus (target vs last price) if model left it empty
    # Do this BEFORE clamping so the fallback can actually trigger
    try:
        # Skip if prompt explicitly declared no consensus
        has_analyst_data = "AnalystConsensus: none" not in prompt
        print(f"[stock] _sanitize: Checking analyst fallback - has_analyst_data={has_analyst_data}")

        if has_analyst_data:
            cm = ctx.get("computed_metrics")
            fund = {}
            last_px = None
            if hasattr(cm, "fundamentals"):
                fund = cm.fundamentals or {}
                last_px = getattr(cm, "last_price", None)
            elif isinstance(cm, dict):
                fund = cm.get("fundamentals") or {}
                last_px = cm.get("last_price")
            target = None
            try:
                target = float(fund.get("analyst_target_price")) if fund.get("analyst_target_price") is not None else None
            except Exception:
                target = None

            # Check if LLM returned a valid score; if not and we have target data, calculate fallback
            llm_score = an.get("score")
            try:
                llm_score = float(llm_score) if llm_score is not None else None
            except (ValueError, TypeError):
                llm_score = None

            print(f"[stock] _sanitize: Analyst fallback values - llm_score={llm_score}, target={target}, last_px={last_px}")

            if llm_score is None and isinstance(target, (int, float)) and isinstance(last_px, (int, float)) and last_px and last_px > 0:
                print(f"[stock] _sanitize: Calculating analyst fallback score...")
                implied = (float(target) / float(last_px)) - 1.0
                score = 50.0 + (implied * 100.0)
                an["score"] = _clamp_score(score)
                print(f"[stock] _sanitize: Using analyst fallback calculation: target={target}, last_px={last_px}, score={an['score']}")
            else:
                print(f"[stock] _sanitize: Analyst fallback skipped - condition failed")
        else:
            print(f"[stock] _sanitize: Analyst fallback skipped - prompt contains 'AnalystConsensus: none'")
    except Exception as e:
        print(f"[stock] _sanitize: Error in analyst fallback calculation: {e}")
        import traceback
        traceback.print_exc()

    # Remove any 'summary' fields from sentiment blocks (LLM should only provide 'score')
    for block in (ns, ss, an):
        if "summary" in block:
            del block["summary"]
        block["score"] = _clamp_score(block.get("score"))

    # Note: We no longer null out news sentiment when news is missing
    # The LLM is now required to ALWAYS provide a numeric score by deriving from other data

    # Social sentiment: prefer FMP data if available, otherwise keep LLM-derived score
    try:
        cm = ctx.get("computed_metrics")
        fund = {}
        if hasattr(cm, "fundamentals"):
            fund = cm.fundamentals or {}
        elif isinstance(cm, dict):
            fund = cm.get("fundamentals") or {}
        soc_raw = fund.get("social_sentiment")

        if soc_raw:
            # FMP data available - use it to override LLM output
            try:
                print(f"[stock] _sanitize: social_sentiment raw keys={list(soc_raw.keys()) if isinstance(soc_raw, dict) else type(soc_raw).__name__}")
            except Exception:
                pass
            score, summary = _derive_social_sentiment_from_fmp(soc_raw)
            ss["score"] = _clamp_score(score)
            # Don't include summary - only score is needed in sentimentDial
            print(f"[stock] _sanitize: socialSentiment from FMP -> score={ss['score']}")
        else:
            # No FMP data - LLM MUST have derived from news (per prompt requirements)
            llm_score = ss.get("score")

            # LLM is required to ALWAYS provide a score, even when deriving from news
            # Remove any 'summary' field that LLM may have incorrectly added
            if "summary" in ss:
                del ss["summary"]

            if llm_score is not None:
                # Keep LLM's derived score and clamp it
                ss["score"] = _clamp_score(llm_score)
                print(f"[stock] _sanitize: using LLM-derived socialSentiment -> score={ss['score']}")
            else:
                # This should never happen if LLM follows instructions, but provide fallback
                print("[stock] _sanitize: WARNING - LLM failed to provide social sentiment score despite requirements")
                ss["score"] = None
    except Exception as e:
        print(f"[stock] _sanitize: error deriving social_sentiment: {e}")
        # Don't override if LLM provided a score
        if ss.get("score") is not None:
            ss["score"] = _clamp_score(ss.get("score"))


    # Competitor P/E coherence checks
    try:
        comps = out.get("aiCompetitorAnalysis")
        if isinstance(comps, list) and comps:
            cm = ctx.get("computed_metrics")
            last_price = None
            if hasattr(cm, "last_price"):
                last_price = cm.last_price
            elif isinstance(cm, dict):
                last_price = cm.get("last_price")
            cleaned = []
            drop_all = False
            for block in comps:
                if not isinstance(block, dict):
                    continue
                metric = str(block.get("metric") or "").strip()
                peers = block.get("peers") or {}
                # Normalize peers numeric values
                norm_peers = {}
                price_like = False
                for k, v in (peers.items() if isinstance(peers, dict) else []):
                    try:
                        fv = float(v)
                        if fv <= 0:
                            continue
                        if last_price and abs(fv - float(last_price)) / float(last_price) <= 0.10:
                            price_like = True
                        norm_peers[k] = fv
                    except Exception:
                        # Skip non-numeric
                        continue
                if metric.upper() == "P/E" and price_like:
                    drop_all = True
                    break
                if metric.upper() == "P/E" and not norm_peers:
                    # Nothing coherent
                    continue
                new_block = dict(block)
                if metric.upper() == "P/E":
                    new_block["peers"] = norm_peers
                cleaned.append(new_block)
            if drop_all:
                out["aiCompetitorAnalysis"] = []
            else:
                out["aiCompetitorAnalysis"] = cleaned
    except Exception:
        pass

    # FINAL STEP: Rebuild sentimentDial with ONLY the correct three keys, no extras
    def _to_int_score_0_100(v, fallback=50):
        """Convert to integer score 0-100, with fallback for None values."""
        try:
            if v is None:
                return fallback
            iv = int(round(float(v)))
            if iv < 0:
                iv = 0
            if iv > 100:
                iv = 100
            return iv
        except Exception:
            return fallback

    out["sentimentDial"] = {
        "newsSentiment": {"score": _to_int_score_0_100(ns.get("score"), fallback=50)},
        "socialSentiment": {"score": _to_int_score_0_100(ss.get("score"), fallback=50)},
        "analystSentiment": {"score": _to_int_score_0_100(an.get("score"), fallback=50)}
    }
    print(f"[stock] _sanitize: Final sentimentDial = {out['sentimentDial']}")

    return out


def _post_openrouter_chat(payload: dict, timeout: int, retries: int = 2, backoff: float = 1.0) -> dict:
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    # Optional helpful headers for OpenRouter analytics
    referer = os.getenv("OPENROUTER_SITE_URL") or os.getenv("HTTP_REFERER")
    if referer:
        headers["HTTP-Referer"] = referer
    app_name = os.getenv("OPENROUTER_APP_NAME") or os.getenv("X_TITLE")
    if app_name:
        headers["X-Title"] = app_name

    last_ex = None
    last_status = None
    last_text = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=timeout)
            last_status = r.status_code
            try:
                last_text = r.text
            except Exception:
                last_text = None
            r.raise_for_status()
            return r.json()
        except Exception as ex:
            last_ex = ex
            if attempt < retries:
                sleep_for = backoff * (2 ** attempt) + random.uniform(0.1, 0.3)
                try:
                    time.sleep(sleep_for)
                except Exception:
                    time.sleep(backoff)
    detail = f"status={last_status}, body={last_text[:500] if isinstance(last_text, str) else last_text}"
    raise RuntimeError(f"OpenRouter chat failed after retries: {last_ex} | {detail}")


def make_llm_openrouter(system_msg_override: str | None = None):
    """Factory function to create an LLM caller for OpenRouter.

    Pass a custom system message via system_msg_override to tailor behavior per endpoint (e.g., portfolio vs. stock).
    """
    model = _get_openrouter_model()
    config = _get_openrouter_config()
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.getenv("OPENROUTER_SITE_URL", ""),
        "X-Title": os.getenv("OPENROUTER_APP_NAME", "Hivest API")
    }

    default_system_msg = (
        "You are a precise JSON generator for single-asset stock analysis.\n"
        "Rules: JSON object only; use only provided DATA; do not invent comps or analysts;\n"
        "If no comps provided, output aiCompetitorAnalysis=[]; if no analyst consensus, set analystSentiment.score=null "
        "and summary='unavailable'. Prefer null/'unavailable' for unknown fields; clamp any sentiment scores into 0..100 if present."
    )
    system_msg = system_msg_override or default_system_msg

    print(f"[llm] OpenRouter client initialized with model={model}")

    def _call(prompt: str) -> str:
        """Makes a call to the OpenRouter API."""
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt}
            ],
            "temperature": config["temperature"],
            "max_tokens": 4096,
            # Ask for strict JSON from compatible providers/models; ignored otherwise
            "response_format": {"type": "json_object"}
        }

        for attempt in range(3):  # Simple retry logic
            try:
                print(f"[llm] Sending prompt to OpenRouter (attempt {attempt + 1}): model={model}, temp={config['temperature']}, timeout={config['timeout']}s, prompt_len={len(prompt)}")
                response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=config["timeout"])

                # Log response status
                print(f"[llm] OpenRouter response: status={response.status_code}")

                response.raise_for_status()
                data = response.json()

                # Log the full response structure for debugging
                print(f"[llm] OpenRouter response keys: {list(data.keys())}")

                # Validate response structure
                if "choices" not in data:
                    print(f"[llm] ERROR: Response missing 'choices' key. Response: {json.dumps(data, indent=2)}")
                    raise ValueError("Invalid OpenRouter response: missing 'choices'")

                if not data["choices"]:
                    print(f"[llm] ERROR: 'choices' array is empty. Response: {json.dumps(data, indent=2)}")
                    raise ValueError("Invalid OpenRouter response: empty 'choices' array")

                choice = data["choices"][0]
                if "message" not in choice:
                    print(f"[llm] ERROR: Choice missing 'message' key. Choice: {json.dumps(choice, indent=2)}")
                    raise ValueError("Invalid OpenRouter response: choice missing 'message'")

                message = choice["message"]
                if "content" not in message:
                    print(f"[llm] ERROR: Message missing 'content' key. Message: {json.dumps(message, indent=2)}")
                    raise ValueError("Invalid OpenRouter response: message missing 'content'")

                content = message["content"]
                if content is None:
                    print(f"[llm] ERROR: Content is None. Full response: {json.dumps(data, indent=2)}")
                    raise ValueError("Invalid OpenRouter response: content is None")

                content = content.strip()
                print(f"[llm] Received response from OpenRouter (chars={len(content)})")

                if len(content) == 0:
                    print(f"[llm] WARNING: Empty content received. Full response: {json.dumps(data, indent=2)}")

                return content
            except requests.exceptions.RequestException as e:
                print(f"[llm] OpenRouter call failed (attempt {attempt + 1}): {e}")
                try:
                    print(f"[llm] Response text: {response.text[:500]}")
                except:
                    pass
                if attempt < 2:
                    time.sleep((2 ** attempt) + random.uniform(0.5, 1.0))  # Exponential backoff
                else:
                    raise RuntimeError(f"OpenRouter chat failed after retries: {e}") from e
            except (KeyError, ValueError, TypeError) as e:
                print(f"[llm] OpenRouter response parsing failed (attempt {attempt + 1}): {e}")
                if attempt < 2:
                    time.sleep((2 ** attempt) + random.uniform(0.5, 1.0))
                else:
                    raise RuntimeError(f"OpenRouter response parsing failed after retries: {e}") from e
        return ""  # Should not be reached

    return _call


def _validate_required_fields(parsed: dict, ctx: dict) -> tuple[bool, list[str]]:
    """Validate that required fields are present and not empty in the LLM output.

    Args:
        parsed: The parsed JSON output from the LLM
        ctx: The analysis context containing input data (news_items, etc.)

    Returns:
        (is_valid, missing_fields): True if all required fields are present and non-empty,
                                    False otherwise with a list of missing/empty field names
    """
    missing = []

    # Check aiDailyBrief (must exist and not be empty string)
    ai_brief = parsed.get("aiDailyBrief")
    if not ai_brief or not isinstance(ai_brief, str) or not ai_brief.strip():
        missing.append("aiDailyBrief")

    # Check sentimentDial structure
    sentiment_dial = parsed.get("sentimentDial")
    if not isinstance(sentiment_dial, dict):
        missing.append("sentimentDial")
    else:
        # Check newsSentiment
        news_sent = sentiment_dial.get("newsSentiment")
        if not isinstance(news_sent, dict) or "score" not in news_sent or news_sent.get("score") is None:
            missing.append("sentimentDial.newsSentiment")

        # Check socialSentiment
        social_sent = sentiment_dial.get("socialSentiment")
        if not isinstance(social_sent, dict) or "score" not in social_sent or social_sent.get("score") is None:
            missing.append("sentimentDial.socialSentiment")

        # Check analystSentiment
        analyst_sent = sentiment_dial.get("analystSentiment")
        if not isinstance(analyst_sent, dict) or "score" not in analyst_sent or analyst_sent.get("score") is None:
            missing.append("sentimentDial.analystSentiment")

    # Check aiSummary.newsSummary with smart validation
    ai_summary = parsed.get("aiSummary")
    if not isinstance(ai_summary, dict):
        missing.append("aiSummary")
    else:
        news_summary = ai_summary.get("newsSummary")
        if news_summary is None:
            missing.append("aiSummary.newsSummary")
        else:
            # Smart validation: if news_items were provided but newsSummary is empty, it's invalid
            news_items = ctx.get("news_items") or []
            if isinstance(news_summary, list) and len(news_summary) == 0 and len(news_items) > 0:
                missing.append("aiSummary.newsSummary (empty but {} news items provided)".format(len(news_items)))

    is_valid = len(missing) == 0
    return is_valid, missing


# --- Flask app replicating hivest/api.py but using OpenRouter ---

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """Endpoint for portfolio analysis."""
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    data = request.get_json(silent=True) or {}
    if 'portfolio' not in data:
        return jsonify({"error": "Request must be JSON with a 'portfolio' field."}), 400

    portfolio_string = data.get('portfolio') or ''
    positions = []
    for holding in str(portfolio_string).split():
        try:
            symbol, weight_str = holding.split(':')
            positions.append({"symbol": symbol.strip(), "weight": float(weight_str)})
        except (ValueError, TypeError):
            pass  # Skip malformed pairs

    if not positions:
        return jsonify({"error": "No valid 'TICKER:WEIGHT' pairs found."}), 400

    try:
        pi = PortfolioInput(holdings=holdings_from_user_positions(positions), timeframe_label="ytd")
        options = PortfolioAnalysisOptions(include_news=True, news_limit=6)

        analysis_context = analyze_portfolio(pi, options)
        prompt = build_portfolio_prompt(
            analysis_context["portfolio_input"],
            analysis_context["computed_metrics"],
            analysis_context["news_items"]
        )

        # Build a deterministic fallback result if LLM is unavailable or disabled
        def _portfolio_fallback_json():
            summary = None
            if _portfolio_fallback_render:
                try:
                    summary = _portfolio_fallback_render(
                        analysis_context["portfolio_input"],
                        analysis_context["computed_metrics"],
                        analysis_context["news_items"],
                    )
                except Exception:
                    summary = None
            summary = summary or "Analysis summary unavailable (LLM disabled). Metrics attached."
            return {
                "mode": "fallback",
                "summary": summary,
                "timeframe": analysis_context["portfolio_input"].timeframe_label,
                "holdings": _to_jsonable(analysis_context["portfolio_input"].holdings),
                "metrics": _to_jsonable(analysis_context["computed_metrics"]),
                "news": _to_jsonable(analysis_context.get("news_items", [])),
                "prompt": prompt,
            }

        no_llm = bool(str(data.get('no_llm') or request.args.get('no_llm') or '').strip().lower() in ('1', 'true', 'yes'))
        if no_llm or not _is_openrouter_configured():
            return jsonify(_portfolio_fallback_json())

        portfolio_system_msg = (
            "You are a precise JSON generator for portfolio analysis.\n"
            "Rules: Output only one JSON object; use only the provided DATA section; "
            "follow the 'Output a JSON object with the following exact keys' in the user message; "
            "do not add extra keys; do not invent facts; prefer concise, decision-useful language."
        )
        llm = make_llm_openrouter(system_msg_override=portfolio_system_msg)
        raw_result = llm(prompt)

        parsed = _extract_json_object(raw_result)
        if parsed is None:
            return jsonify({"error": "LLM did not return a valid JSON object.", "raw": raw_result}), 502
        return jsonify(parsed)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500


@app.route('/stock-analysis', methods=['POST', 'OPTIONS'])
def stock_analysis():
    """Endpoint for single-stock analysis."""
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    data = request.get_json(silent=True) or {}

    symbol = (data.get('symbol') or data.get('ticker') or '').strip().upper()
    if not symbol:
        return jsonify({"error": "Request must be JSON with a 'symbol' (or 'ticker') field."}), 400

    timeframe = str(data.get('timeframe') or data.get('timeframe_label') or 'ytd').strip()

    # Options
    include_news = True if 'include_news' not in data else bool(data.get('include_news'))
    news_limit = data.get('news_limit') or 6
    try:
        news_limit = int(news_limit)
    except Exception:
        news_limit = 6
    include_events = True if 'include_events' not in data else bool(data.get('include_events'))

    try:
        # Debug flag handling
        debug_flag = bool(str(data.get('debug') or request.args.get('debug') or '').strip().lower() in ('1', 'true', 'yes'))

        print(f"[stock] Request received: symbol={symbol}, timeframe={timeframe}, include_news={include_news}, news_limit={news_limit}, include_events={include_events}, debug={debug_flag}")

        si = StockInput(symbol=symbol, timeframe_label=timeframe)
        options = StockAnalysisOptions(include_news=include_news, news_limit=news_limit, include_events=include_events, verbose=debug_flag)
        print("[stock] Created StockInput and AnalysisOptions")

        print("[stock] Starting analyze_stock...")
        _t0 = time.time()
        ctx = analyze_stock(si, options)
        _dt = time.time() - _t0
        print(f"[stock] analyze_stock completed in {_dt:.2f}s")

        # News and events info
        news_items = ctx.get("news_items") or []
        events = ctx.get("upcoming_events") or []
        print(f"[stock] News fetched: {len(news_items)} items; Events fetched: {len(events)} items")
        try:
            for i, n in enumerate(news_items[:3]):
                title = (n.get("title") or n.get("headline") or "") if isinstance(n, dict) else str(n)
                date = (n.get("date") or n.get("publishedAt") or n.get("time")) if isinstance(n, dict) else ""
                print(f"[stock]   News[{i}]: {title[:120]} | {date}")
        except Exception:
            pass

        # Social sentiment presence in fundamentals
        try:
            cm = ctx.get("computed_metrics")
            fund = {}
            if hasattr(cm, "fundamentals"):
                fund = cm.fundamentals or {}
            elif isinstance(cm, dict):
                fund = cm.get("fundamentals") or {}
            soc_raw = fund.get("social_sentiment")
            if soc_raw:
                keys = list(soc_raw.keys()) if isinstance(soc_raw, dict) else type(soc_raw).__name__
                print(f"[stock] Social sentiment raw available with keys: {keys}")
            else:
                print("[stock] Social sentiment raw not available in fundamentals")
        except Exception:
            print("[stock] Social sentiment inspection failed")

        print("[stock] Building stock prompt...")
        prompt = build_stock_prompt(ctx)
        print(f"[stock] Prompt built (length={len(prompt)} chars)")

        # Debug logging (model, temp, symbol, prompt excerpt)
        try:
            import os as _os
            dbg_env = _os.getenv('HIVEST_DEBUG_STOCK', '0').strip()
            if debug_flag or dbg_env == '1':
                model = _get_openrouter_model()
                cfg = _get_openrouter_config()
                head = prompt[:1000]
                tail = prompt[-1000:] if len(prompt) > 1000 else ''
                print(f"[stock-debug] model={model} temp={cfg['temperature']} symbol={symbol} timeframe={timeframe}")
                print(f"[stock-debug] PROMPT_HEAD:\n{head}")
                if tail:
                    print(f"[stock-debug] PROMPT_TAIL:\n{tail}")
        except Exception:
            pass

        # Fallback JSON if LLM is disabled/unavailable
        def _stock_fallback_json():
            return {
                "mode": "fallback",
                "note": "Analysis summary unavailable (LLM disabled). Metrics and data attached.",
                "symbol": ctx["stock_input"].symbol,
                "timeframe": ctx["stock_input"].timeframe_label,
                "metrics": _to_jsonable(ctx["computed_metrics"]),
                "news": _to_jsonable(ctx.get("news_items", [])),
                "events": _to_jsonable(ctx.get("upcoming_events", [])),
                "prompt": prompt,
            }

        no_llm = bool(str(data.get('no_llm') or request.args.get('no_llm') or '').strip().lower() in ('1', 'true', 'yes'))
        configured = _is_openrouter_configured()
        print(f"[stock] no_llm={no_llm}, openrouter_configured={configured}")
        if no_llm or not configured:
            print("[stock] Using fallback JSON (LLM disabled or not configured)")
            return jsonify(_stock_fallback_json())

        print("[stock] Initializing OpenRouter LLM client...")
        llm = make_llm_openrouter()

        # Retry loop for handling non-JSON responses and missing required fields
        max_retries = 2  # Total of 3 attempts (initial + 2 retries)
        parsed = None
        raw_result = None
        retry_reason = None

        for attempt in range(max_retries + 1):
            if attempt > 0:
                retry_delay = (2 ** (attempt - 1)) + random.uniform(0.1, 0.5)
                print(f"[stock] Retry attempt {attempt}/{max_retries} after {retry_delay:.2f}s delay due to {retry_reason}")
                time.sleep(retry_delay)

            print(f"[stock] Calling OpenRouter with prompt (attempt {attempt + 1}/{max_retries + 1})...")
            raw_result = llm(prompt)
            print(f"[stock] OpenRouter call completed, raw_result_len={len(raw_result) if isinstance(raw_result, str) else 'n/a'}")

            parsed = _extract_json_object(raw_result)
            if parsed is None:
                print(f"[stock] Failed to extract JSON object from LLM response (attempt {attempt + 1}/{max_retries + 1})")
                retry_reason = "JSON parsing failure"
                if attempt < max_retries:
                    print(f"[stock] Will retry with same prompt...")
                continue

            print(f"[stock] Successfully extracted JSON object on attempt {attempt + 1}")

            # Validate required fields
            is_valid, missing_fields = _validate_required_fields(parsed, ctx)
            if is_valid:
                print(f"[stock] All required fields present and valid on attempt {attempt + 1}")
                break
            else:
                print(f"[stock] Required fields missing or empty on attempt {attempt + 1}: {', '.join(missing_fields)}")
                retry_reason = f"missing fields: {', '.join(missing_fields)}"
                if attempt < max_retries:
                    print(f"[stock] Will retry with same prompt...")
                    parsed = None  # Reset to trigger retry

        if parsed is None:
            print(f"[stock] Failed to get valid response after {max_retries + 1} attempts")
            return jsonify({"error": "LLM did not return a valid JSON object with all required fields after retries.", "raw": raw_result, "attempts": max_retries + 1, "last_reason": retry_reason}), 502

        sanitized = _sanitize_stock_output(ctx, prompt, parsed)
        print("[stock] Sanitization complete; responding with JSON")

        # Optional JSON dump for dev harness
        try:
            import os as _os
            if _os.getenv('HIVEST_STOCK_DUMP_JSON', '0').strip() == '1':
                dbg_dir = _os.path.join(PROJECT_ROOT, '.debug')
                _os.makedirs(dbg_dir, exist_ok=True)
                fname = _os.path.join(dbg_dir, f"stock_analysis_{symbol}.json")
                with open(fname, 'w', encoding='utf-8') as f:
                    json.dump(sanitized, f, indent=2)
        except Exception:
            pass

        return jsonify(sanitized)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
