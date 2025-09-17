import os, requests, datetime as _dt
from typing import Dict


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "hivest-shared/1.0"})
    return s


def get_fundamentals(symbol: str) -> Dict[str, float]:
    """
    Try FMP first, optionally Alpha Vantage later.
    Returns a small dict like: {'pe_ttm': float, 'market_cap': float, 'price': float, 'rev_cagr': float}
    Missing values are simply omitted.
    """
    sym = symbol.upper().strip()
    out: Dict[str, float] = {}
    sess = _session()

    fmp = (os.getenv("FMP_FREE_API_KEY") or "").strip()
    if fmp:
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/quote/{sym}",
                         params={"apikey": fmp}, timeout=7)
            if r.ok and isinstance(r.json(), list) and r.json():
                q = r.json()[0]
                if q.get("pe") is not None: out["pe_ttm"] = float(q["pe"])
                if q.get("price") is not None: out["price"] = float(q["price"])
                if q.get("marketCap") is not None: out["market_cap"] = float(q["marketCap"])
        except Exception: pass
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}",
                         params={"limit": 4, "apikey": fmp}, timeout=7)
            if r.ok and isinstance(r.json(), list) and r.json():
                rows = r.json()
                revs = [float(x["revenue"]) for x in rows if x.get("revenue") is not None]
                if len(revs) >= 2 and revs[0] and revs[-1]:
                    years = max(1, len(revs)-1)
                    out["rev_cagr"] = (revs[0] / revs[-1])**(1/years) - 1.0
        except Exception: pass

    # Alpha Vantage optional extension later (respect free-tier limits)
    return out
