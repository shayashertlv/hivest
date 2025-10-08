"""hivest/shared/news.py"""
"""
Ollama-based position brief helpers.
Generates concise, headline-like briefs per holding using local Ollama, with inputs:
- stock ticker (symbol)
- percentage of the portfolio (weight fraction 0..1)
- average buying price (avg_buy_price)
- when the stock was bought (bought_at, optional YYYY-MM-DD)

Returns items shaped like {source, title, url, publishedAt} for compatibility.
If Ollama isn't available, returns an empty list.
"""
from typing import List, Dict, Optional, Union
from datetime import datetime
import os
import requests
from typing import Iterable
from .llm_client import make_llm

SymbolLike = Union[str, List[str], None]


def _flatten(items: SymbolLike) -> List[str]:
    if items is None:
        return []
    if isinstance(items, str):
        return [items]
    return [s for s in items if isinstance(s, str)]

def _norm_syms(symbols: Iterable[str]) -> list[str]:
    return [str(s).upper().strip() for s in (symbols or []) if str(s).strip()]

def fetch_news_api(symbols: list[str], limit: int = 10) -> list[dict]:
    """
    Fetch stock news from Financial Modeling Prep (FMP) only.
    This version now also fetches full-text press releases.

    Output shape: [{source, title, url, publishedAt, symbol, content}]
    """
    syms = _norm_syms(symbols)
    if not syms:
        return []

    fmp = (os.getenv("FMP_FREE_API_KEY") or "").strip()
    if not fmp:
        return []

    session = requests.Session()
    session.headers.update({"User-Agent": "hivest-news/1.0"})

    out: list[dict] = []

    # 1. Fetch press releases (they have full content)
    for symbol in syms:
        try:
            r = session.get(
                f"https://financialmodelingprep.com/api/v3/press-releases/{symbol}",
                params={"limit": limit, "apikey": fmp},
                timeout=7,
            )
            if r.ok and isinstance(r.json(), list):
                for a in r.json():
                    out.append({
                        "source": "Press Release",
                        "title": a.get("title") or "",
                        "url": f"https://financialmodelingprep.com/press-releases/{symbol}", # Placeholder URL
                        "publishedAt": a.get("date") or "",
                        "symbol": (a.get("symbol") or "").upper(),
                        "content": a.get("text") or "" # Full content is in the 'text' field
                    })
        except Exception:
            pass # Continue if one symbol fails

    # 2. Fetch standard news (headlines and metadata)
    tickers = ",".join(sorted(set(syms)))
    api_limit = min(50, max(limit, min(5, limit) * len(syms)))

    try:
        r = session.get(
            "https://financialmodelingprep.com/api/v3/stock_news",
            params={"tickers": tickers, "limit": api_limit, "apikey": fmp},
            timeout=7,
        )
        if r.ok and isinstance(r.json(), list):
            for a in r.json():
                sym = (a.get("symbol") or "").upper()
                if sym and sym not in syms:
                    continue
                out.append({
                    "source": a.get("site") or "FMP News",
                    "title": a.get("title") or "",
                    "url": a.get("url") or "",
                    "publishedAt": a.get("publishedDate") or "",
                    "symbol": sym,
                    "content": a.get("text") or "" # Add content, though it's usually empty for this endpoint
                })
    except Exception:
        pass

    # Sort by date and return the most recent articles, up to the limit.
    out = [item for item in out if item.get("content", "").strip()]
    out.sort(key=lambda x: x.get("publishedAt", ""), reverse=True)
    return out[: max(0, limit * len(syms)) or limit]

def get_news_for_holdings(holdings: list[dict], limit: int = 10) -> list[dict]:
    """
    Preferred entry point:
    1) Try web news via FMP for the portfolio symbols.
    2) If none available, fallback to local Ollama-based one-liners via build_position_brief().

    When trimming to `limit`, interleave items by symbol (round-robin) to avoid
    one ticker dominating when the limit is small.
    """
    syms = [str(h.get("symbol", "")).upper() for h in (holdings or []) if h.get("symbol")]
    api_items = fetch_news_api(syms, limit=limit)
    if api_items:
        try:
            # Round-robin by symbol
            buckets: dict[str, list[dict]] = {}
            order: list[str] = []
            for it in api_items:
                sym = (it.get('symbol') or '').upper()
                if sym not in buckets:
                    buckets[sym] = []
                    order.append(sym)
                buckets[sym].append(it)
            out: list[dict] = []
            # Iterate buckets in order repeatedly until limit reached
            while len(out) < limit and buckets:
                progressed = False
                for sym in list(order):
                    q = buckets.get(sym, [])
                    if q:
                        out.append(q.pop(0))
                        progressed = True
                        if len(out) >= limit:
                            break
                    if not q:
                        # remove empty bucket from cycle
                        buckets.pop(sym, None)
                        order = [s for s in order if s in buckets]
                if not progressed:
                    break
            return out
        except Exception:
            return api_items[:limit]
    return build_position_brief(holdings, limit=limit) or []

def build_position_brief(holdings: List[Dict], limit: int = 10) -> List[Dict]:
    """
    Build short, headline-like briefs for the provided holdings using Ollama.
    holdings: list of dicts with keys: symbol, weight, avg_buy_price, bought_at (optional)
    """
    llm = make_llm()
    if not callable(llm):
        print("[news] Ollama unavailable -> returning empty briefs.")
        return []
    else:
        print("[news] Using Ollama to draft position briefs.")

    now = datetime.now().strftime("%Y-%m-%d")
    lines = []
    for h in holdings or []:
        sym = str(h.get("symbol", "")).upper()
        if not sym:
            continue
        w = float(h.get("weight") or 0.0)
        abp = h.get("avg_buy_price")
        bat = h.get("bought_at") or "unknown date"
        lines.append(f"- {sym}: weight={w:.2%}; avg_buy_price={abp if abp is not None else 'n/a'}; bought_at={bat}")

    if not lines:
        return []

    prompt = (
        "You are a cautious financial assistant. Based ONLY on the provided positions, "
        "produce up to {limit} concise, single-line briefs (no numbering) that a holder should monitor. "
        "Avoid making up facts or targets; keep generic and risk-aware. Each line should start with the ticker.\n\n"
        "Positions:\n{positions}\n\n"
        "Output: a list of short headline-like lines, one per consideration."
    ).format(limit=limit, positions="\n".join(lines))

    text = (llm(prompt) or "").strip()
    # Split into lines and sanitize
    out: List[Dict] = []
    for ln in (text.splitlines() if text else []):
        s = ln.strip("- •* \t")
        if not s:
            continue
        # extract leading ticker token if present (e.g., "AAPL: ...", "MSFT — ...")
        lead = s.split(":", 1)[0].split("—", 1)[0].split("-", 1)[0].strip().split()[0]
        sym_guess = "".join(ch for ch in lead if ch.isalnum()).upper()
        item = {
            "source": "ollama",
            "title": s,
            "url": "",
            "publishedAt": now,
            "symbol": sym_guess if sym_guess else "",
        }
        out.append(item)
        if len(out) >= limit:
            break

    return out


def get_briefs_for_symbols(symbols: SymbolLike = None, sectors: SymbolLike = None, limit: int = 10) -> List[Dict]:
    """
    Compatibility helper: if symbols provided, generate generic briefs with Ollama.
    """
    syms = _flatten(symbols)
    if not syms:
        return []
    holdings = [{"symbol": s, "weight": 0.0, "avg_buy_price": None, "bought_at": None} for s in syms]
    return build_position_brief(holdings, limit=limit)