hivest/api.py – Simple API README

Endpoint
- URL: POST /analyze
- Content-Type: application/json

Request body
- Required field: portfolio (string)
- Format: space-separated pairs of TICKER:WEIGHT
  - TICKER: stock symbol (e.g., AAPL, MSFT)
  - WEIGHT: number (float), e.g., 0.5
- Example: {"portfolio": "AAPL:0.5 MSFT:0.3 XOM:0.2"}

Behavior
- The server parses the portfolio string into positions like [{"symbol": "AAPL", "weight": 0.5}, ...].
- Invalid pairs (not in SYMBOL:WEIGHT format or non-numeric weights) are ignored.
- If no valid positions remain, the request is rejected.

Responses
- 200 OK
  - Body: {"analysis": string}
  - analysis is an LLM-generated string based on the provided portfolio context. It may look like JSON text, but it is returned as a string.

- 400 Bad Request
  - Missing portfolio: {"error": "Invalid input. 'portfolio' key is missing."}
  - No valid holdings: {"error": "No valid holdings found in the portfolio string."}

Examples
- Start the dev server (from project root):
  python -m hivest.api

- Quick test (single-line curl):
  curl -X POST http://127.0.0.1:5000/analyze -H "Content-Type: application/json" -d "{\"portfolio\":\"AAPL:0.5 MSFT:0.3 XOM:0.2\"}"

- PowerShell (Invoke-RestMethod)
  $body = @{ portfolio = "AAPL:0.5 MSFT:0.3 XOM:0.2" } | ConvertTo-Json
  Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/analyze -ContentType 'application/json' -Body $body

- curl (Windows, multi-line)
  curl -X POST http://127.0.0.1:5000/analyze ^
       -H "Content-Type: application/json" ^
       -d "{\"portfolio\": \"AAPL:0.5 MSFT:0.3 XOM:0.2\"}"

Notes
- When run directly (python hivest/api.py), the Flask dev server listens on http://127.0.0.1:5000.
- Timeframe is fixed to "ytd" inside the endpoint.
