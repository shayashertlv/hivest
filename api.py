from flask import Flask, request, jsonify
from hivest.portfolio_analysis.processing.engine import holdings_from_user_positions, analyze_portfolio
from hivest.portfolio_analysis.processing.models import PortfolioInput, AnalysisOptions
from hivest.portfolio_analysis.llm.prompts import build_portfolio_prompt
from hivest.shared.llm_client import make_llm

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    """
    This is the main API endpoint for portfolio analysis.
    It expects a JSON payload with a "portfolio" key,
    which is a string in the format 'TICKER1:WEIGHT1 TICKER2:WEIGHT2 ...'
    """
    data = request.get_json()
    if not data or 'portfolio' not in data:
        return jsonify({"error": "Invalid input. 'portfolio' key is missing."}), 400

    portfolio_string = data['portfolio']
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