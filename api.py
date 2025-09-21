import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from .portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from .portfolio_analysis.processing.models import PortfolioInput, AnalysisOptions
from .portfolio_analysis.llm.prompts import build_portfolio_prompt
from .shared.llm_client import make_llm

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

@app.route('/analyze', methods=['GET', 'POST', 'OPTIONS'])
def analyze():
    """
    This is the main API endpoint for portfolio analysis.
    It accepts both GET and POST requests.
    - POST: Expects a JSON body -> {"portfolio": "AAPL:0.5 MSFT:0.3"}
    - GET: Expects a URL query parameter -> /analyze?portfolio=AAPL:0.5%20MSFT:0.3
    """
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    portfolio_string = None
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if data and 'portfolio' in data:
            portfolio_string = data.get('portfolio')
    elif request.method == 'GET':
        portfolio_string = request.args.get('portfolio')

    if not portfolio_string or not isinstance(portfolio_string, str) or not portfolio_string.strip():
        return jsonify({
            "error": "The 'portfolio' string is missing or empty. Please provide it in the JSON body for a POST request or as a URL parameter for a GET request.",
            "examples": {
                "GET": "/analyze?portfolio=AAPL:0.5%20MSFT:0.3",
                "POST": {"portfolio": "AAPL:0.5 MSFT:0.3"}
            }
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

        llm = make_llm("llama3:8b")
        analysis_result = llm(prompt)

        return jsonify({"analysis": analysis_result})

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal server error occurred during analysis."}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)