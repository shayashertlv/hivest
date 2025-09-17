from datetime import datetime
from typing import Dict

from ..processing.models import StockMetrics


def _fmt_fundamentals(items: Dict[str, float]) -> str:
    if not items:
        return "none"
    return ", ".join(f"{k}={v:.2f}" for k, v in items.items())


def build_stock_prompt(symbol: str, tf_label: str, m: StockMetrics) -> str:
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    perf = {
        "cum_return_pct": f"{m.cum_return*100:.2f}",
        "volatility_pct": f"{m.volatility*100:.2f}",
        "beta_vs_spy": f"{m.beta_vs_spy:.2f}",
        "max_drawdown_pct": f"{m.max_drawdown*100:.2f}",
    }
    technicals = {
        "rsi14": f"{m.rsi14:.1f}",
        "sma20": f"{m.sma20:.2f}",
        "sma50": f"{m.sma50:.2f}",
        "sma200": f"{m.sma200:.2f}",
        "pct_from_52w_high": f"{m.pct_from_52w_high*100:.2f}",
        "pct_from_52w_low": f"{m.pct_from_52w_low*100:.2f}",
    }
    fundamentals = _fmt_fundamentals(m.fundamentals)
    news = "\n".join(
        f"- {it.get('title','')} [source={it.get('source','')}; date={(it.get('publishedAt','') or '')[:10]}]"
        for it in (m.news_items or [])[:6]
    ) or "none"
    nxt = m.next_earnings or "none"

    instruction = (
        "You are an equity research analyst preparing a forward-looking brief.\n\n"
        "Output format:\n"
        "📊 Company Snapshot\n"
        "<Single line: SYMBOL (Company) → concise description including business model, strategic focus, and most recent revenue/earnings trajectory.>\n"
        "<Optional second line covering scale/capex if relevant.>\n"
        "\n"
        "⚖️ Key Advantages\n"
        "- 3-4 bullets describing competitive positioning, secular demand, balance sheet strength, or execution momentum. Each bullet should lean on provided metrics or news (cite sources by name, e.g., Reuters).\n"
        "\n"
        "⚠️ Risks to Watch\n"
        "- 3 bullets flagging valuation stretch, competitive threats, regulatory overhang, or execution/capex risks.\n"
        "\n"
        "🔮 12–18-Month Outlook (what could drive upside/downside)\n"
        "- 3-4 bullets focusing strictly on upcoming catalysts, adoption curves, margin levers, or macro signposts to monitor.\n"
        "Include the next earnings checkpoint if available.\n"
        "\n"
        "📝 Conclusions & Recommendations\n"
        "- 2-3 bullets summarising investment stance (accumulate/hold/trim), key execution priorities, and valuation thoughts.\n"
        "\n"
        "💡 Bottom line: <one-sentence forward-looking call summarising the thesis and key swing factors>.\n\n"
        "Rules:\n"
        "* Stay grounded in the supplied fundamentals, technicals, and news.\n"
        "* Emphasise how future demand, monetisation, or capacity plans impact the story.\n"
        "* Never fabricate numbers or dates."
    )

    payload = (
        f"Generated: {dt_now}\n"
        f"Symbol: {symbol.upper()}\n"
        f"Timeframe: {tf_label}\n"
        f"Performance: {perf}\n"
        f"Technicals: {technicals}\n"
        f"Fundamentals: {fundamentals}\n"
        f"NextEarnings: {nxt}\n"
        f"News:\n{news}\n"
    )
    return instruction + "\n\n--- DATA ---\n" + payload
