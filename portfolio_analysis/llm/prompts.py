"""hivest/portfolio_analysis/llm/prompts.py"""
from datetime import datetime
from typing import List, Dict, Optional, Tuple

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
    symbols, sectors, weights = _holdings_to_maps(pi.holdings)
    ppy = float(getattr(pi, "periods_per_year", 252.0)) or 252.0
    insights = derive_insights(cm, ppy=ppy)
    score_info = score_portfolio(cm)

    flags_line = " ".join(insights.get("flags", [])) or "none"
    pre_actions = ", ".join(insights.get("actions", [])) or "none"
    dt_now = datetime.now().strftime("%Y-%m-%d %H:%M")

    port_perf_s = f"Portfolio: {FMT_PCT(cm.cum_return)}"
    spy_perf_s = f"SPY: {FMT_PCT(cm.benchmarks.get('SPY', {}).get('cum_return', 0.0))}" if 'SPY' in cm.benchmarks else ""
    qqq_perf_s = f"QQQ: {FMT_PCT(cm.benchmarks.get('QQQ', {}).get('cum_return', 0.0))}" if 'QQQ' in cm.benchmarks else ""
    comparison_summary_line = ", ".join(filter(None, [port_perf_s, spy_perf_s, qqq_perf_s]))

    # Positions block (exact facts only)
    pos_lines = []
    for h in (pi.holdings or []):
        sym = (h.symbol or "").upper()
        w = weights.get(sym, float(h.weight or 0.0))
        abp = h.avg_buy_price
        bat = h.bought_at or "unknown"
        name = h.name or sym # Use symbol if name is missing
        pos_lines.append(f"- {name} ({sym}): weight={w:.2%}; avg_buy_price={abp if abp is not None else 'n/a'}; bought_at={bat}")
    positions_block = "\n".join(pos_lines) or "n/a"

    winners_s = _fmt_top_list(cm.winners, 3) or "n/a"
    losers_s  = _fmt_top_list(cm.losers, 3) or "n/a"

    sym_cum = _per_symbol_cum_map(pi)
    stock_perf_block = ", ".join(f"{s}: {FMT_PCT(v)}" for s, v in sorted(sym_cum.items())) or "n/a"

    # Identify top-2 by weight and surface their cumulative returns explicitly
    h_by_w = sorted(((h.symbol.upper(), float(getattr(h, "weight", 0.0))) for h in (pi.holdings or [])), key=lambda t: t[1], reverse=True)
    from_syms = [s for s, _ in h_by_w[:2]]
    top2_s = ", ".join(f"{s}: {FMT_PCT(sym_cum.get(s, 0.0))}" for s in from_syms) or "n/a"

    sector_top_s    = _fmt_top_list(cm.top_sectors, 2) or "n/a"
    sector_bottom_s = _fmt_top_list(cm.bottom_sectors, 2) or "n/a"

    bench_lines = []
    for name, obj in (cm.benchmarks or {}).items():
        b_cum = obj.get("cum_return", 0.0)
        rel   = obj.get("relative_vs_portfolio", 0.0)
        bench_lines.append(f"- {name}: cum={FMT_PCT(b_cum)}, relative_vs_portfolio={FMT_PCT(rel)}")
    bench_block = "\n".join(bench_lines) or "n/a"

    conc = cm.concentration or {}

    drag_sym, drag_contrib = (None, 0.0)
    if cm.losers:
        drag_sym, drag_contrib = cm.losers[0]
    drag_weight = float(weights.get(drag_sym or "", 0.0))
    drag_is_material = (drag_sym is not None) and (drag_contrib < 0.0) and (drag_weight >= 0.10 or abs(drag_contrib) >= 0.01)

    risk_block = (
        f"volatility={FMT_SIG(cm.volatility)}; "
        f"volatility_pct={FMT_PCT(cm.volatility)}; "
        f"beta={FMT_SIG(cm.beta)}; "
        f"sharpe={FMT_SIG(cm.sharpe)}; "
        f"sortino={FMT_SIG(cm.sortino)}; "
        f"max_drawdown={FMT_PCT(cm.max_drawdown)}; "
        f"hhi={conc.get('hhi', 0):.4f}; "
        f"effective_n={conc.get('effective_n', 0):.2f}; "
        f"top_weight={conc.get('top_weight', 0.0):.2%}; "
        f"warnings={'; '.join(conc.get('warnings', [])) or 'none'}"
    )

    # News block (titles and content)
    nitems = news_items or []
    news_block = "\n".join(
        f"- {it.get('symbol','')}: {it.get('title','')} [source={it.get('source','')}; date={(it.get('publishedAt','') or '')[:10]}]\nContent: {it.get('content', 'N/A')}"
        for it in nitems
    ) or "none"

    include_events = bool(getattr(pi, "include_events", True))
    events_src = (pi.upcoming_events or []) if include_events else []
    events_block = "\n".join(
        f"- {ev.get('symbol','')}: {ev.get('type','event')} on {ev.get('date','')} {ev.get('note','')}".strip()
        for ev in events_src[:3]
    ) or "none"

    instruction = (
        "You are a direct, insightful senior portfolio manager. Your task is to generate a structured JSON object providing a sharp, "
        "comprehensive analysis of an investment portfolio. Your tone should be that of an experienced human analyst, avoiding generic phrasing "
        "and repetitive sentence structures. **Do not output any text, code, or explanations outside of the final JSON object.**\n\n"

        "Only write the conclusions. For example, use 'consider adding another holding in order to create a better diversification for a steadier portfolio' "
        "instead of 'adding a small diversifier to raise effective N'.\n\n"

        "You will be provided with quantitative data (weights, performance figures) and a small set of news. "
        "Your goal is to synthesize these into a qualitative, strategic analysis that goes beyond simply stating the obvious.\n\n"

        "**Before writing, perform these internal checks (do not output them):** "
        "if both weight and recent_return are provided, compute a simple contribution proxy per holding = weight × recent_return; "
        "use this to identify top drivers and to align recommendations with the stated risks. "
        "Identify the single largest weight and note if it exceeds a soft threshold (e.g., 30–35%) to assess concentration risk. "
        "Do not invent any numbers not supplied.\n\n"

        "**Output a JSON object with the following exact keys:**\n\n"

        "1.  \"portfolioScore\": (Number) A single overall quality score for the portfolio from 1 (poor) to 10 (excellent), "
        "based on a synthesis of all provided data (performance, risk, diversification).\n\n"

        "2.  \"portfolioAllocation\": (Array of Objects) Detail each holding with an insightful summary:\n"
        "    -  HoldingName — **Write two sentences that provide a sharp, data-rich narrative.** The first sentence must state the holding's weight and recent performance. The second sentence must connect that performance to a specific business driver, market trend, or a key insight from the provided news. **Go beyond the obvious and provide a 'so what' for the investor.**\n\n"

        "3. \"News\": (Array of Strings) **Return between 2 and 4 of the most impactful news items** that a portfolio owner should see. Use your judgment: if the provided articles are not significant, you may return fewer than 2. "
        "Each item MUST describe a specific event or action from the supplied news. "
        "Do not write generic commentary or predictions. "
        "Write each item as a single, self-contained sentence in sentence case that names the holding, states what happened, ties it to a clear financial mechanism (e.g., revenue/pricing/margins/opex/guidance), and states the directional impact. "
        "End the sentence with exactly ' (+)' for positive or ' (-)' for negative. "
        "Do not copy or lightly edit headlines.\n"
        " - Rely only on information provided in the prompt. No outside knowledge.\n"
        " - Omit any item that cannot be tied to a financial mechanism or lacks a concrete event.\n"
        " - If multiple items concern the same holding, include only the most material one.\n\n"

        "4.  \"strategicRecommendations\": (String) **Return a single, plain-English, 1–2 sentence conclusion** that includes: "
        "(a) the biggest portfolio risk, (b) one forward-looking opportunity, and (c) one specific step to improve resilience or quality. "
        "When the identified risk is concentration, **prefer** to reference the specific oversized holding in the action (e.g., 'trim <HoldingName>'). "
        "**Avoid quant jargon and metrics** and ensure the recommendation does not contradict the diagnosed risk.\n\n"

        "**General rules:**\n"
        "- Use only information supplied in the prompt.\n"
        "- Keep writing tight and specific; avoid boilerplate and clichés.\n"
        "- All keys above are required; if a section has no qualifying content, return an empty array or empty string.\n"
        "- Do **not** add extra keys.\n"
        "- Return **only** the final JSON object."
    )

    payload = (
      f"Timeframe: {pi.timeframe_label}\n"
      f"Generated: {dt_now}\n"
      f"Symbols: {', '.join(symbols)}\n"
      f"Sectors: {', '.join(sectors)}\n\n"
      f"Positions:\n{positions_block}\n\n"
      f"Overall cumulative return: {FMT_PCT(cm.cum_return)}\n"
      f"Benchmarks:\n{bench_block}\n"
      f"Winners (top): {winners_s}\n"
      f"Laggards (top): {losers_s}\n"
      f"Top-2 by weight cum returns: {top2_s}\n"
      f"DragCandidate: symbol={drag_sym or 'none'}, contrib={FMT_PCT(drag_contrib)}, weight={FMT_PCT(drag_weight)}, is_material={drag_is_material}\n"
      f"Sectors up: {sector_top_s}\n"
      f"Sectors down: {sector_bottom_s}\n"
      f"Upcoming events: \n{events_block}\n"
      f"News (titles and content):\n{news_block}\n"
      f"Pre-Suggested Actions: {pre_actions}\n"
      f"Per-symbol cumulative returns: {stock_perf_block}\n"
      f"PerformanceSummaryLine: {comparison_summary_line}\n"
      f"Score hint: {score_info['score']}/10 — {score_info['reasons']}\n"
    )
    return instruction + "\n\n--- DATA ---\n" + payload


def _fallback_render(pi: PortfolioInput, cm: ComputedMetrics, news_items: Optional[List[Dict]] = None) -> str:
    # Deterministic fallback: single engaging paragraph with salient facts only.
    ppy = float(getattr(pi, "periods_per_year", 252.0)) or 252.0
    ins = derive_insights(cm, ppy=ppy)

    bench_parts: List[str] = []
    for name, obj in (cm.benchmarks or {}).items():
        b_cum = float(obj.get("cum_return", 0.0))
        rel = float(obj.get("relative_vs_portfolio", 0.0))
        vibe = "ahead of" if rel > 0 else "behind" if rel < 0 else "in line with"
        bench_parts.append(f"{name} {FMT_PCT(b_cum)} ({vibe} by {FMT_PCT(abs(rel))})")
    bench_text = ", ".join(bench_parts) or "no benchmarks"

    def _cap(items: List[Tuple[str, float]]) -> str:
        return ", ".join(f"{sym} {FMT_PCT(val)}" for sym, val in (items[:2] if items else [])) or "none"

    winners_s = _cap(cm.winners)
    losers_s = _cap(cm.losers)
    sector_up = _cap(cm.top_sectors)
    sector_down = _cap(cm.bottom_sectors)

    evs = pi.upcoming_events or []
    ev_text = "; ".join(
        f"{ev.get('symbol','')}: {ev.get('type','event')} {ev.get('date','')}".strip()
        for ev in evs[:2]
    ) or "no imminent events"

    nws = news_items or []
    news_text = "; ".join((it.get('title','') or '').strip() for it in nws[:2]) or "no recent briefs"

    hhi = float((cm.concentration or {}).get("hhi", 0.0))
    topw = float((cm.concentration or {}).get("top_weight", 0.0))
    if hhi < 0.15:
        conc_phrase = "highly diversified"
    elif hhi < 0.25:
        conc_phrase = "moderately diversified"
    else:
        conc_phrase = "concentrated"

    spy_cum = None
    for k, obj in (cm.benchmarks or {}).items():
        if k.upper() == "SPY":
            spy_cum = float(obj.get("cum_return", 0.0))
            break
    numbers = [f"portfolio {FMT_PCT(float(cm.cum_return))}"]
    if spy_cum is not None:
        numbers.append(f"SPY {FMT_PCT(spy_cum)}")
    numbers.append(f"top weight {topw:.0%}")

    final_action = (" " + ins["actions"][0]) if ins.get("actions") else ""

    from .prompts import score_portfolio as _score  # local reuse, avoid circular
    score_info = _score(cm)

    return (
        f"Over {pi.timeframe_label}, {', '.join(numbers)}; {bench_text}. "
        f"Leaders: {winners_s}; laggards: {losers_s}. "
        f"Sectors up: {sector_up}; down: {sector_down}. "
        f"The portfolio is {conc_phrase}, so single-name concentration is the main swing factor. "
        f"Upcoming: {ev_text}. Briefs: {news_text}.{final_action} "
        f"Score: {int(score_info['score'])}/10 — {score_info['reasons']}."
    )