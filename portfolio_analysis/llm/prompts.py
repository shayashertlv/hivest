from datetime import datetime
from typing import Dict, List, Optional, Tuple

from ..processing.models import PortfolioInput, ComputedMetrics, Holding
from ...shared.performance import cumulative_return


FMT_PCT = lambda x: f"{x*100:.2f}%"
FMT_SIG = lambda x: f"{x:.4f}"


def _fmt_top_list(items: List[Tuple[str, float]], k: int) -> str:
    pairs = items[:k]
    return ", ".join(f"{sym}: {FMT_PCT(val)}" for sym, val in pairs)


def _holdings_to_maps(holdings: List[Holding]) -> Tuple[List[str], List[str], Dict[str, float]]:
    symbols = [h.symbol.upper() for h in (holdings or [])]
    sectors = list({(h.sector or "Unknown") for h in (holdings or [])})
    weights = {h.symbol.upper(): float(h.weight or 0.0) for h in (holdings or [])}
    return symbols, sectors, weights


def _per_symbol_cum_map(pi: PortfolioInput) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for sym, rets in (pi.per_symbol_returns or {}).items():
        try:
            out[sym] = cumulative_return(rets)
        except Exception:
            out[sym] = 0.0
    return out


def derive_insights(cm: "ComputedMetrics", ppy: float = 252.0) -> dict:
    """
    Turn raw stats into simple 'flags' and prioritized 'actions'.
    We annualize Sharpe/Sortino internally so callers don't need to modify cm.
    """
    # Defensive fetches
    conc = cm.concentration or {}
    hhi = float(conc.get("hhi", 0.0))
    effn = float(conc.get("effective_n", 0.0))
    topw = float(conc.get("top_weight", 0.0))

    # Annualize (assume daily unless caller passes a different ppy)
    sharpe_ann = float(cm.sharpe) * (ppy ** 0.5)
    sortino_ann = float(cm.sortino) * (ppy ** 0.5)

    flags: list[str] = []
    actions: list[str] = []

    all_green = all((v >= 0.0) for _, v in (cm.by_symbol or []))
    if all_green:
        # If everyone is contributing, avoid recommending trims purely on concentration.
        actions = [a for a in actions if not a.lower().startswith("rebalance: trim")]
        # Prefer non-disruptive breadth
        actions.insert(0, "Maintain weights; if you want more breadth, add a small diversifier with fresh capital.")

    # Simple thresholds (tune to taste)
    conc_hi = (hhi >= 0.25) or (effn and effn < 5) or (topw >= 0.25)
    sharpe_bad = sharpe_ann < 0.5
    sharpe_strong = sharpe_ann >= 1.0
    sortino_edge = (sortino_ann - sharpe_ann) > 0.3
    beta_high = float(cm.beta) >= 1.2
    dd_severe = float(cm.max_drawdown) <= -0.20
    dd_mod = float(cm.max_drawdown) <= -0.10

    if conc_hi: flags.append("[Concentration risk]")
    if sharpe_bad: flags.append("[Low Sharpe]")
    if sharpe_strong: flags.append("[High Sharpe]")
    if sortino_edge: flags.append("[Upside-skewed: Sortino >> Sharpe]")
    if beta_high: flags.append("[High Beta]")
    if dd_severe: flags.append("[Severe drawdown]")
    elif dd_mod: flags.append("[Moderate drawdown]")

    # Prioritized actions
    if conc_hi:
        actions.append("Rebalance: trim the largest weight(s) and add a diversifier to raise effective N.")
    if sharpe_bad and beta_high:
        actions.append("De-risk: lower beta or add a small hedge (e.g., index put spread / inverse ETF slice).")
    if sortino_edge and not sharpe_bad:
        actions.append("Maintain exposure but add a stop or trailing hedge to guard downside shocks.")
    if cm.losers:
        laggard, _ = cm.losers[0]
        actions.append(f"Review laggard: {laggard} underperformed; trim if conviction is weak.")

    # Deduplicate and cap to top-3
    actions = list(dict.fromkeys(actions))[:3]
    return {
        "sharpe_annual": sharpe_ann,
        "sortino_annual": sortino_ann,
        "flags": flags,
        "actions": actions,
    }


def score_portfolio(cm: "ComputedMetrics") -> Dict[str, str | float | int]:
    # Bench outperformance: average relative vs benchmarks
    rels = [v.get("relative_vs_portfolio", 0.0) for v in (cm.benchmarks or {}).values()]
    avg_rel = sum(rels) / max(1, len(rels))
    # Base 5; add up to +3 for outperformance, +2 for risk quality; subtract up to -2 for drawdown/concentration
    score = 5.0
    score += max(0.0, min(3.0, (avg_rel * 100.0) / 10.0))
    sortino_edge = cm.sortino_annual - cm.sharpe_annual
    if sortino_edge > 0.3: score += 0.5
    if cm.max_drawdown > -0.15: score += 1.0
    if (cm.concentration or {}).get("effective_n", 10) < 4.0:
        score -= 1.0
    if (cm.concentration or {}).get("top_weight", 0.0) >= 0.40:
        score -= 1.0
    score = max(1.0, min(10.0, round(score)))
    reasons = []
    if avg_rel > 0.0: reasons.append("clear outperformance vs benchmarks")
    if cm.max_drawdown > -0.15: reasons.append("downside kept contained")
    if ((cm.concentration or {}).get("effective_n", 10) < 4.0): reasons.append("concentration limits breadth")
    if len(reasons) < 2: reasons.append("risk/return trade-off reasonable")
    return {"score": int(score), "reasons": "; ".join(reasons[:2])}


def build_portfolio_prompt(pi: PortfolioInput, cm: ComputedMetrics, news_items: Optional[List[Dict]] = None) -> str:
    _, _, weights = _holdings_to_maps(pi.holdings)
    ppy = float(getattr(pi, "periods_per_year", 252.0)) or 252.0
    insights = derive_insights(cm, ppy=ppy)
    score_info = score_portfolio(cm)

    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")
    window_display = getattr(pi, "timeframe_display", pi.timeframe_label)
    analysis_window = f"{getattr(pi, 'analysis_start', 'unknown')} to {getattr(pi, 'analysis_end', 'unknown')}"

    sym_cum = _per_symbol_cum_map(pi)
    contrib_map = {sym: val for sym, val in (cm.by_symbol or [])}

    holdings_lines = []
    for h in sorted((pi.holdings or []), key=lambda x: float(getattr(x, "weight", 0.0)), reverse=True):
        sym = (h.symbol or "").upper()
        weight = weights.get(sym, float(h.weight or 0.0))
        holdings_lines.append(
            f"- {sym}: weight={weight:.2%}; name={getattr(h, 'name', '') or 'n/a'}; "
            f"sector={(h.sector or 'Unknown')}; ytd_cum={FMT_PCT(sym_cum.get(sym, 0.0))}; "
            f"contribution={FMT_PCT(contrib_map.get(sym, 0.0))}"
        )
    holdings_block = "\n".join(holdings_lines) or "none"

    bench_lines = []
    for name, obj in sorted((cm.benchmarks or {}).items()):
        bench_lines.append(
            f"- {name}: cum={FMT_PCT(obj.get('cum_return', 0.0))}; "
            f"relative_vs_portfolio={FMT_PCT(obj.get('relative_vs_portfolio', 0.0))}"
        )
    bench_block = "\n".join(bench_lines) or "none"

    news_items = news_items or []
    news_block = "\n".join(
        f"- {it.get('symbol','')}: {it.get('title','')} [source={it.get('source','')}; date={(it.get('publishedAt','') or '')[:10]}]"
        for it in news_items[:8]
    ) or "none"

    sector_block = ", ".join(f"{sec}: {FMT_PCT(val)}" for sec, val in (cm.by_sector or [])[:6]) or "none"
    risk_summary = {
        "volatility_pct": FMT_PCT(cm.volatility),
        "beta": FMT_SIG(cm.beta),
        "max_drawdown": FMT_PCT(cm.max_drawdown),
        "sharpe": FMT_SIG(cm.sharpe),
        "sortino": FMT_SIG(cm.sortino),
        "hhi": f"{(cm.concentration or {}).get('hhi', 0.0):.4f}",
        "effective_n": f"{(cm.concentration or {}).get('effective_n', 0.0):.2f}",
        "top_weight": f"{(cm.concentration or {}).get('top_weight', 0.0):.2%}",
        "warnings": "; ".join((cm.concentration or {}).get('warnings', [])) or "none",
    }

    winners_s = _fmt_top_list(cm.winners, 3) or "n/a"
    losers_s = _fmt_top_list(cm.losers, 3) or "n/a"

    instruction = (
        "You are a seasoned portfolio strategist. Draft a forward-looking briefing for an investor "
        "using ONLY the supplied data. The analysis must always be framed as Year-to-Date (YTD).\n\n"
        "Format the response EXACTLY with the following sections and spacing:\n"
        "📊 Portfolio Allocation\n"
        "<one line per holding in descending weight, no bullets. Format each line as 'SYMBOL (Company Name if available) – XX% → short forward-looking comment'. "
        "Blend weight, sector role, and contribution facts. Mention sector balance and strategic role (growth, defensive, income, etc.).\n"
        "\n"
        "⚖️ Key Advantages\n"
        "- Three to five concise bullets about strengths, diversification, quality, secular themes, or risk-adjusted outperformance.\n"
        "\n"
        "⚠️ Risks to Watch\n"
        "- Two to four bullets highlighting concentration, macro/sector exposures, laggards, valuation stretch, or drawdown considerations.\n"
        "\n"
        "📝 Conclusions & Recommendations\n"
        "- Two to three bullets summarising overall quality, balance, and specific diversification or allocation moves to consider.\n"
        "\n"
        "Close with one sentence starting with '💡 Bottom line:' that synthesises the go-forward stance.\n\n"
        "Guidance:\n"
        "* Keep all numbers in percent with whole numbers when possible (no more than one decimal).\n"
        "* Reference benchmark context, risk stats, winners/laggards, and sector mix explicitly.\n"
        "* Use the news feed to anchor catalysts or secular themes (cite the source name in plain text where relevant).\n"
        "* Do not invent data beyond what is provided."
    )

    payload = (
        f"Generated: {dt_now}\n"
        f"WindowDisplay: {window_display}\n"
        f"AnalysisWindow: {analysis_window}\n"
        f"PortfolioCumReturn: {FMT_PCT(cm.cum_return)}\n"
        f"WinnersTop: {winners_s}\n"
        f"LosersTop: {losers_s}\n"
        f"Score: {score_info['score']} ({score_info['reasons']})\n"
        f"InsightsFlags: {' '.join(insights.get('flags', [])) or 'none'}\n"
        f"InsightsActions: {', '.join(insights.get('actions', [])) or 'none'}\n"
        f"Holdings:\n{holdings_block}\n"
        f"Sectors: {sector_block}\n"
        f"Benchmarks:\n{bench_block}\n"
        f"Risk: {risk_summary}\n"
        f"News:\n{news_block}\n"
    )

    return instruction + "\n\n--- DATA ---\n" + payload


def _fallback_render(pi: PortfolioInput, cm: ComputedMetrics, news_items: Optional[List[Dict]] = None) -> str:
    # Deterministic fallback aligned with the structured template.
    ppy = float(getattr(pi, "periods_per_year", 252.0)) or 252.0
    insights = derive_insights(cm, ppy=ppy)
    score_info = score_portfolio(cm)

    sym_cum = _per_symbol_cum_map(pi)
    contrib_map = {sym: val for sym, val in (cm.by_symbol or [])}
    holdings_sorted = sorted((pi.holdings or []), key=lambda h: float(getattr(h, "weight", 0.0)), reverse=True)

    allocation_lines: List[str] = []
    for h in holdings_sorted:
        sym = (h.symbol or "").upper()
        weight_pct = float(getattr(h, "weight", 0.0)) * 100
        sector = h.sector or "Unknown"
        perf = sym_cum.get(sym, 0.0)
        contrib = contrib_map.get(sym, 0.0)
        trend = "driving gains" if contrib > 0 else "tempering returns" if contrib < 0 else "holding steady"
        allocation_lines.append(
            f"{sym} ({getattr(h, 'name', '') or sector}) – {weight_pct:.0f}% → {sector} exposure {trend}; YTD move {FMT_PCT(perf)}."
        )

    benchmarks = cm.benchmarks or {}
    bench_snippets = []
    for name, obj in benchmarks.items():
        rel = float(obj.get("relative_vs_portfolio", 0.0))
        gap_phrase = "ahead of" if rel < 0 else "lagging" if rel > 0 else "in line with"
        bench_snippets.append(f"{name} {gap_phrase} by {FMT_PCT(abs(rel))}")
    bench_summary = ", ".join(bench_snippets) or "no benchmark context"

    sectors = {h.sector or "Unknown" for h in (pi.holdings or [])}
    winners = ", ".join(f"{sym} {FMT_PCT(val)}" for sym, val in (cm.winners or [])[:2]) or "n/a"
    losers = ", ".join(f"{sym} {FMT_PCT(val)}" for sym, val in (cm.losers or [])[:2]) or "n/a"

    advantages = [
        f"YTD return {FMT_PCT(cm.cum_return)} with {bench_summary}.",
        f"Breadth across {', '.join(sorted(sectors))}.",
        f"Leaders: {winners}."
    ]

    risks: List[str] = []
    top_weight = float((cm.concentration or {}).get("top_weight", 0.0))
    if top_weight >= 0.25:
        risks.append(f"Top position at {top_weight*100:.0f}% concentrates returns.")
    if float(cm.beta) > 1.1:
        risks.append(f"Beta {FMT_SIG(cm.beta)} > 1 suggests amplified market swings.")
    if losers != "n/a":
        risks.append(f"Watch laggards: {losers}.")
    if not risks:
        risks.append("Stay alert to macro shocks despite balanced mix.")

    recommendations: List[str] = []
    actions = insights.get("actions", [])
    if actions:
        recommendations.append(actions[0])
    if len(actions) > 1:
        recommendations.append(actions[1])
    recommendations.append(f"Score {score_info['score']}/10 — {score_info['reasons']}.")

    bottom_line = (
        "💡 Bottom line: Maintain discipline around the YTD leaders while nibbling on complementary sectors to keep the mix "
        "resilient."
    )

    return (
        "📊 Portfolio Allocation\n"
        + "\n".join(allocation_lines or ["No holdings provided."])
        + "\n\n⚖️ Key Advantages\n"
        + "\n".join(f"- {ln}" for ln in advantages)
        + "\n\n⚠️ Risks to Watch\n"
        + "\n".join(f"- {ln}" for ln in risks)
        + "\n\n📝 Conclusions & Recommendations\n"
        + "\n".join(f"- {ln}" for ln in recommendations)
        + f"\n\n{bottom_line}"
    )
