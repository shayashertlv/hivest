from flask import Flask, request, jsonify
from .portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from .portfolio_analysis.processing.models import PortfolioInput, AnalysisOptions
from .portfolio_analysis.llm.prompts import build_portfolio_prompt
from .shared.llm_client import make_llm


app = Flask(__name__)

@app.route('/analyze', methods=['GET', 'POST', 'OPTIONS'])
def analyze():
    """
    This is the main API endpoint for portfolio analysis.
    It accepts:
    - POST with JSON body: {"portfolio": "AAPL:0.5 MSFT:0.3 GOOG:0.2"}
    - GET with either a query parameter ?portfolio=... or a JSON body (if provided by the client)
    The portfolio value is a string 'TICKER1:WEIGHT1 TICKER2:WEIGHT2 ...'
    """
    data = request.get_json(silent=True) or {}
    portfolio_string = data.get('portfolio') or request.args.get('portfolio')
    if not portfolio_string:
        return jsonify({
            "error": "Invalid input. Provide 'portfolio' as JSON body or query parameter.",
            "examples": {
                "GET": "/analyze?portfolio=AAPL:0.5 MSFT:0.3 GOOG:0.2",
                "POST": {"portfolio": "AAPL:0.5 MSFT:0.3 GOOG:0.2"}
            }
        }), 400

    positions = []
    # Parse the portfolio string into a list of dictionaries
    for holding in portfolio_string.split():
        try:
            symbol, weight_str = holding.split(':')
            weight = float(weight_str)
            positions.append({"symbol": symbol, "weight": weight})
        except ValueError:
            # You can decide to either fail here or just skip the invalid ones
            pass

    if not positions:
        return jsonify({"error": "No valid holdings found in the portfolio string."}), 400

    # The following is adapted from your main.py's run_from_user_input function
    timeframe = "ytd"
    holdings = holdings_from_user_positions(positions)

    pi = PortfolioInput(
        holdings=holdings,
        timeframe_label=timeframe,
        # The following will be auto-filled by auto_fill_series
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

    # Return the analysis as a JSON response
    return jsonify({"analysis": analysis_result})

if __name__ == '__main__':
    # This will run a development server on http://127.0.0.1:5000
    app.run(debug=True)