import os
import time
import random
import traceback
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

from hivest.portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from hivest.portfolio_analysis.processing.models import PortfolioInput, AnalysisOptions
from hivest.portfolio_analysis.llm.prompts import build_portfolio_prompt


# --- OpenRouter LLM helper (replicates llm_client.make_llm flow, but for OpenRouter) ---

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_openrouter_model(passed: str | None = None) -> str:
    # Priority: explicit param -> env -> default mapping similar to llama3:8b
    if passed and isinstance(passed, str) and passed.strip():
        model_in = passed.strip()
    else:
        model_in = (os.getenv("OPENROUTER_MODEL", "").strip() or "")

    if not model_in:
        # Default to a close analogue of llama3:8b instruct on OpenRouter
        return "meta-llama/llama-3.1-8b-instruct"

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


def make_llm_openrouter(model_name: str | None = None):
    model = _get_openrouter_model(model_name)
    timeout = _get_openrouter_timeout()
    temperature = _get_openrouter_temperature()

    def _call(prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a helpful financial analyst."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1024,
            "stream": False,
        }

        data = _post_openrouter_chat(payload, timeout, retries=2, backoff=1.0)
        # Standard OpenRouter response shape
        try:
            choices = data.get("choices", [])
            if not choices:
                return ""
            msg = choices[0].get("message", {}) or {}
            return (msg.get("content") or "").strip()
        except Exception:
            return ""

    return _call


# --- Flask app replicating hivest/api.py but using OpenRouter ---

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """
    This is the main API endpoint for portfolio analysis.
    It accepts only POST requests with a JSON body.
    - POST: Expects a JSON body -> {"portfolio": "AAPL:0.5 MSFT:0.3"}
    """
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    if not request.is_json:
        return jsonify({
            "error": "Content-Type must be application/json. Provide the 'portfolio' field in the JSON body.",
            "example": {"portfolio": "AAPL:0.5 MSFT:0.3"}
        }), 400

    data = request.get_json(silent=True) or {}
    portfolio_string = data.get('portfolio') if isinstance(data, dict) else None

    if not portfolio_string or not isinstance(portfolio_string, str) or not portfolio_string.strip():
        return jsonify({
            "error": "The 'portfolio' string is missing or empty in the JSON body.",
            "example": {"portfolio": "AAPL:0.5 MSFT:0.3"}
        }), 400

    positions = []
    # Parse the portfolio string into a list of dictionaries
    for holding in portfolio_string.split():
        try:
            symbol, weight_str = holding.split(':')
            weight = float(weight_str)
            positions.append({"symbol": symbol.strip(), "weight": weight})
        except (ValueError, TypeError):
            # Skip any malformed pairs
            pass

    if not positions:
        return jsonify({"error": "No valid holdings were found. Please ensure the format is 'TICKER:WEIGHT'."}), 400

    try:
        timeframe = "ytd"
        holdings = holdings_from_user_positions(positions)

        pi = PortfolioInput(
            holdings=holdings,
            timeframe_label=timeframe,
            portfolio_returns=[],
            benchmark_returns={},
            market_returns=[],
            per_symbol_returns={},
            risk_free_rate=0.0001,
            upcoming_events=[],
        )

        options = AnalysisOptions(
            include_news=True,
            news_limit=6,
        )

        analysis_context = analyze_portfolio(pi, options)
        prompt = build_portfolio_prompt(
            analysis_context["portfolio_input"],
            analysis_context["computed_metrics"],
            analysis_context["news_items"]
        )

        # Use OpenRouter LLM instead of Ollama
        llm = make_llm_openrouter("openai/gpt-oss-120b")
        analysis_result = llm(prompt)

        return jsonify({"analysis": analysis_result})

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred during analysis."}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
