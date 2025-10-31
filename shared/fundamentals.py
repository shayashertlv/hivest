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

        # --- Step 3: Fetch last earnings surprise (for quarterly bottom-line) ---
        try:
            print(f"[fundamentals] Fetching earnings surprises for {sym}...")
            r = sess.get(f"https://financialmodelingprep.com/api/v3/earnings-surprises/{sym}", params={"limit": 1, "apikey": fmp}, timeout=7)
            print(f"[fundamentals] Earnings surprises response: status={r.status_code}, ok={r.ok}")
            if r.ok and isinstance(r.json(), list) and r.json():
                print(f"[fundamentals] Earnings surprises data length: {len(r.json())}")
                row = r.json()[0] or {}
                # Common field names seen in FMP payloads (try multiple variations)
                actual = (
                    row.get("actualEPS") if row.get("actualEPS") is not None else
                    row.get("actualEarningResult") if row.get("actualEarningResult") is not None else
                    row.get("eps")
                )
                estimate = (
                    row.get("estimatedEPS") if row.get("estimatedEPS") is not None else
                    row.get("estimatedEarning") if row.get("estimatedEarning") is not None else
                    row.get("epsEstimated")
                )
                try:
                    actual_f = float(actual) if actual is not None else None
                except Exception:
                    actual_f = None
                try:
                    est_f = float(estimate) if estimate is not None else None
                except Exception:
                    est_f = None
                surprise_pct = None
                try:
                    sp = row.get("surprisePercentage")
                    surprise_pct = float(sp) if sp is not None else None
                except Exception:
                    surprise_pct = None
                bottom = None
                if actual_f is not None and est_f is not None:
                    if actual_f > est_f * 1.01:
                        bottom = "beat"
                    elif actual_f < est_f * 0.99:
                        bottom = "miss"
                    else:
                        bottom = "in line"
                if bottom or (actual_f is not None or est_f is not None):
                    out["earnings"] = {
                        "date": row.get("date") or row.get("period") or None,
                        "actualEPS": actual_f,
                        "estimatedEPS": est_f,
                        "surprisePct": surprise_pct,
                        "bottomLine": bottom,
                    }
                    print(f"[fundamentals] Earnings data populated: bottomLine={bottom}, actualEPS={actual_f}, estimatedEPS={est_f}")
                else:
                    print(f"[fundamentals] Earnings data not populated: bottom={bottom}, actual_f={actual_f}, est_f={est_f}")
            else:
                print(f"[fundamentals] Earnings surprises: empty or invalid response")
        except Exception as e:
            print(f"[fundamentals] Error fetching earnings surprises: {e}")

    else:  # Instrument is ETF
        # --- Step 5: Fetch ETF Holdings ---
        try:
            r = sess.get(f"https://financialmodelingprep.com/api/v3/etf-holder/{sym}", params={"apikey": fmp}, timeout=7)
            if r.ok and isinstance(r.json(), list) and r.json():
                etf_data["top_holdings"] = [{"symbol": h.get("asset"), "weight": h.get("weightPercentage")} for h in r.json()[:5]]
        except Exception:
            pass

    # --- Fetch Social Sentiment (for ALL instrument types) ---
    try:
        print(f"[fundamentals] Fetching social sentiment for {sym}...")
        r = sess.get(f"https://financialmodelingprep.com/api/v4/social-sentiment", params={"symbol": sym, "apikey": fmp}, timeout=7)
        print(f"[fundamentals] Social sentiment response: status={r.status_code}, ok={r.ok}, content_length={len(r.content)}")

        # Try to parse JSON regardless of status to see what we got
        try:
            social_data = r.json()
            print(f"[fundamentals] Social sentiment raw JSON: {social_data}")
        except Exception as json_err:
            print(f"[fundamentals] Social sentiment: failed to parse JSON - {json_err}")
            print(f"[fundamentals] Social sentiment raw text (first 500 chars): {r.text[:500]}")
            social_data = None

        if r.ok and social_data:
            print(f"[fundamentals] Social sentiment data type: {type(social_data).__name__}")
            if isinstance(social_data, list):
                print(f"[fundamentals] Social sentiment data length: {len(social_data)}")
                if social_data:
                    # If the API returns a list, take the first item
                    out["social_sentiment"] = social_data[0]
                    print(f"[fundamentals] Social sentiment keys: {list(social_data[0].keys()) if isinstance(social_data[0], dict) else 'not a dict'}")
                else:
                    print(f"[fundamentals] Social sentiment: empty list returned")
            elif isinstance(social_data, dict):
                out["social_sentiment"] = social_data
                print(f"[fundamentals] Social sentiment keys: {list(social_data.keys())}")
            else:
                print(f"[fundamentals] Social sentiment: unexpected data structure")
        else:
            if not r.ok:
                print(f"[fundamentals] Social sentiment: HTTP error {r.status_code}")
            elif social_data is None:
                print(f"[fundamentals] Social sentiment: null/empty JSON response")
            elif isinstance(social_data, list) and len(social_data) == 0:
                print(f"[fundamentals] Social sentiment: empty list (no data for {sym})")
            else:
                print(f"[fundamentals] Social sentiment: falsy response - {social_data}")
    except Exception as e:
        print(f"[fundamentals] Error fetching social sentiment: {e}")
        import traceback
        traceback.print_exc()

    # --- Fetch Analyst Price Targets (for ALL instrument types) ---
    try:
        print(f"[fundamentals] Fetching analyst price target consensus for {sym}...")
        r = sess.get(f"https://financialmodelingprep.com/api/v4/price-target-consensus", params={"symbol": sym, "apikey": fmp}, timeout=7)
        print(f"[fundamentals] Price target consensus response: status={r.status_code}, ok={r.ok}")
        if r.ok and isinstance(r.json(), list) and r.json():
            data = r.json()
            print(f"[fundamentals] Price target consensus data length: {len(data)}")
            if data:
                item = data[0]
                print(f"[fundamentals] Price target consensus keys: {list(item.keys())}")
                # FMP v4 price-target-consensus returns: targetHigh, targetLow, targetConsensus, targetMedian
                target_price = item.get("targetConsensus") or item.get("targetMedian")
                print(f"[fundamentals] targetConsensus = {target_price}")
                if target_price is not None:
                    out["analyst_target_price"] = float(target_price)
        else:
            print(f"[fundamentals] Price target consensus: empty or invalid response")
    except Exception as e:
        print(f"[fundamentals] Error fetching price target consensus: {e}")

    return inst_type, out, etf_data
