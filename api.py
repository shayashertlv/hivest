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


@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    """
    This is the main API endpoint for portfolio analysis.
    It accepts POST requests with a JSON body:
    {"portfolio": "AAPL:0.5 MSFT:0.3 GOOG:0.2"}
    """
    if request.method == 'OPTIONS':
        return jsonify(success=True), 200

    # This endpoint now only accepts POST requests.
    if request.method == 'POST':
        data = request.get_json(silent=True)
        if not data or 'portfolio' not in data:
            return jsonify({
                "error": "Invalid POST request. 'portfolio' key is missing from the JSON body.",
                "example": {"portfolio": "AAPL:0.5 MSFT:0.3 GOOG:0.2"}
            }), 400

        portfolio_string = data.get('portfolio')

        if not isinstance(portfolio_string, str) or not portfolio_string.strip():
            return jsonify({"error": "'portfolio' must be a non-empty string."}), 400

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
            return jsonify({"error": "No valid holdings found. Ensure the format is 'TICKER:WEIGHT'."}), 400

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
            return jsonify({"error": "An internal server error occurred."}), 500

    # If the request is not POST or OPTIONS, this part will not be reached
    # because of the 'methods' argument in the route decorator.
    return jsonify({"error": "Method not allowed."}), 405


if __name__ == '__main__':
    app.run(debug=True, port=5000)