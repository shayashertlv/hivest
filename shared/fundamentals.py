"""hivest/shared/fundamentals.py"""
import os
import requests
from typing import Dict, Tuple, Any


def _session():
    s = requests.Session()
    s.headers.update({"User-Agent": "hivest-shared/1.0"})
    return s


def get_fundamentals(symbol: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """
    Fetches fundamental, social sentiment, and analyst estimate data from FMP.
    Returns a tuple: (instrument_type, fundamentals_dict, etf_profile_dict)
    """
    sym = symbol.upper().strip()
    out: Dict[str, Any] = {}
    etf_data: Dict[str, Any] = {}
    inst_type: str = "stock"
    sess = _session()
    fmp = (os.getenv("FMP_FREE_API_KEY") or "").strip()

    if not fmp:
        return inst_type, out, etf_data

    # --- Step 1: Fetch Profile (to determine instrument type) ---
    try:
        r = sess.get(f"https://financialmodelingprep.com/api/v3/profile/{sym}", params={"apikey": fmp}, timeout=7)
        if r.ok and isinstance(r.json(), list) and r.json():
            profile = r.json()[0]
            if profile.get("isEtf"):
                inst_type = "etf"
                if profile.get("expenseRatio") is not None:
                    etf_data["expense_ratio"] = float(profile["expenseRatio"])
            
            # Get Price, PE, MktCap from Profile to reduce API calls
            if profile.get("price") is not None: out["price"] = float(profile["price"])
            if profile.get("mktCap") is not None: out["market_cap"] = float(profile["mktCap"])
            # FMP does not provide PE on profile, so we'll get it from quotes if needed later
    except Exception:
        pass

    if inst_type == "stock":
        # --- Step 2: Fetch Financial Statements for Stocks ---
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/income-statement/{sym}", params={"period": "annual", "limit": 2, "apikey": fmp}, timeout=7)
            if r.ok and r.json():
                rows = r.json()
                # Get TTM PE from Key Metrics if not available on quote
                r_metrics = sess.get(f"https://financialmodelingprep.com/api/v3/key-metrics-ttm/{sym}", params={"apikey": fmp}, timeout=7)
                if r_metrics.ok and r_metrics.json():
                    out['pe_ttm'] = r_metrics.json()[0].get('peRatioTTM')

                eps_list = [float(x["eps"]) for x in rows if x.get("eps") is not None]
                if len(eps_list) >= 2: out["epsGrowthYoY"] = (eps_list[0] / eps_list[1]) - 1.0 if eps_list[1] else None

                if rows[0].get("grossProfitRatio") is not None:
                    out["grossMargin"] = rows[0].get("grossProfitRatio")

        except Exception:
            pass

        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{sym}", params={"period": "annual", "limit": 1, "apikey": fmp}, timeout=7)
            if r.ok and r.json():
                bs = r.json()[0]
                if bs.get("debtToEquity") is not None:
                    out["debtToEquity"] = bs.get("debtToEquity")
        except Exception:
            pass

        # --- Step 3: Fetch Social Sentiment ---
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v4/social-sentiment", params={"symbol": sym, "apikey": fmp}, timeout=7)
            if r.ok and r.json():
                social_data = r.json()
                if isinstance(social_data, list) and social_data:
                    # If the API returns a list, take the first item
                    out["social_sentiment"] = social_data[0]
                else:
                    out["social_sentiment"] = social_data
        except Exception:
            pass
            
        # --- Step 4: Fetch Analyst Price Targets ---
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/analyst-estimates/{sym}", params={"apikey": fmp}, timeout=7)
            if r.ok and isinstance(r.json(), list) and r.json():
                out["analyst_target_price"] = r.json()[0].get("estimatedPriceAvg")
        except Exception:
            pass

    else:  # Instrument is ETF
        # --- Step 5: Fetch ETF Holdings ---
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/etf-holder/{sym}", params={"apikey": fmp}, timeout=7)
            if r.ok and isinstance(r.json(), list) and r.json():
                etf_data["top_holdings"] = [{"symbol": h.get("asset"), "weight": h.get("weightPercentage")} for h in r.json()[:5]]
        except Exception:
            pass
            
    return inst_type, out, etf_data
