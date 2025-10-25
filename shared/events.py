"""hivest/shared/events.py"""
import os, requests, datetime as _dt
from typing import Optional

def next_earnings_date(symbol: str) -> Optional[str]:
    fmp = (os.getenv("FMP_FREE_API_KEY") or "").strip()
    if not fmp: return None
    try:
        today = _dt.date.today().isoformat()
        fut = (_dt.date.today() + _dt.timedelta(days=365)).isoformat()
        print(f"[events] Fetching earnings calendar for {symbol.upper()} from {today} to {fut}")
        r = requests.get("https://financialmodelingprep.com/api/v3/earning_calendar",
                         params={"symbol": symbol.upper(), "from": today, "to": fut, "apikey": fmp},
                         timeout=7)
        print(f"[events] Earnings calendar response: status={r.status_code}, ok={r.ok}")
        if r.ok and isinstance(r.json(), list) and r.json():
            data = r.json()
            print(f"[events] Earnings calendar returned {len(data)} dates: {[item.get('date') for item in data[:5]]}")
            # Filter for the requested symbol since API returns all companies
            next_date = None
            for item in data:
                if item.get("symbol", "").upper() == symbol.upper():
                    next_date = item.get("date")
                    break
            print(f"[events] Next earnings date selected for {symbol.upper()}: {next_date}")
            return next_date
        else:
            print(f"[events] No earnings calendar data returned")
    except Exception as e:
        print(f"[events] Error fetching earnings calendar: {e}")
        return None
    return None
