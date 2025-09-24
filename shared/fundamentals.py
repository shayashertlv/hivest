"""hivest/shared/fundamentals.py"""
import os, requests, datetime as _dt
from typing import Dict, Tuple


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "hivest-shared/1.0"})
    return s


def get_fundamentals(symbol: str) -> Tuple[str, Dict[str, float]]:
    """
    Try FMP first, optionally Alpha Vantage later.
    Returns a tuple: (instrument_type, fundamentals_dict)
      - instrument_type: "stock" or "etf" (defaults to "stock" on error/unknown)
      - fundamentals_dict: e.g., {'pe_ttm': float, 'market_cap': float, 'price': float, 'rev_cagr': float}
    Missing values are simply omitted.
    """
    sym = symbol.upper().strip()
    out: Dict[str, float] = {}
    inst_type: str = "stock"
    sess = _session()

    fmp = (os.getenv("FMP_FREE_API_KEY") or "").strip()
    if fmp:
        # Fetch real-time quote first (used for both stocks and ETFs)
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/quote/{sym}",
                         params={"apikey": fmp}, timeout=7)
            if r.ok and isinstance(r.json(), list) and r.json():
                q = r.json()[0]
                if q.get("pe") is not None: out["pe_ttm"] = float(q["pe"])
                if q.get("price") is not None: out["price"] = float(q["price"])
                if q.get("marketCap") is not None: out["market_cap"] = float(q["marketCap"])
        except Exception:
            pass

        # Fetch profile to determine instrument type (ETF vs stock)
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/profile/{sym}",
                         params={"apikey": fmp}, timeout=7)
            if r.ok and isinstance(r.json(), list) and r.json():
                p = r.json()[0]
                is_etf_val = p.get("isEtf")
                # Normalize possible string/number/boolean representations
                is_etf = False
                if isinstance(is_etf_val, bool):
                    is_etf = is_etf_val
                elif isinstance(is_etf_val, (int, float)):
                    is_etf = bool(is_etf_val)
                elif isinstance(is_etf_val, str):
                    is_etf = is_etf_val.strip().lower() in {"true", "1", "yes", "y"}
                if is_etf:
                    inst_type = "etf"
        except Exception:
            # Default to stock on any profile error
            pass

        # For stocks, attempt to compute revenue CAGR from income statements
        if inst_type == "stock":
            try:
                r = sess.get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                             params={"limit": 4, "apikey": fmp}, timeout=7)
                if r.ok and isinstance(r.json(), list) and r.json():
                    rows = r.json()
                    revs = [float(x["revenue"]) for x in rows if x.get("revenue") is not None]
                    if len(revs) >= 2 and revs[0] and revs[-1]:
                        years = max(1, len(revs) - 1)
                        out["rev_cagr"] = (revs[0] / revs[-1]) ** (1 / years) - 1.0
            except Exception:
                pass

    # Alpha Vantage optional extension later (respect free-tier limits)
    return inst_type, out
