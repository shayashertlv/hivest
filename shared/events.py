import os, requests, datetime as _dt
from typing import Optional

def next_earnings_date(symbol: str) -> Optional[str]:
    fmp = (os.getenv("FMP_FREE_API_KEY") or "").strip()
    if not fmp: return None
    try:
        today = _dt.date.today().isoformat()
        fut = (_dt.date.today() + _dt.timedelta(days=365)).isoformat()
        r = requests.get("https://financialmodelingprep.com/api/v3/earning_calendar",
                         params={"symbol": symbol.upper(), "from": today, "to": fut, "apikey": fmp},
                         timeout=7)
        if r.ok and isinstance(r.json(), list) and r.json():
            return r.json()[0].get("date") or None
    except Exception:
        return None
    return None
