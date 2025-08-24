# Top-5 Options Breakout Scanner (Starter App)

This is a **starter** FastAPI app that scans a 10-ticker watchlist and suggests top 5 options day trades.
It uses **Polygon.io** for market and options data (you must provide an API key).

## Features
- Background scanner that polls polygon for 1-min bars and option snapshots
- Detects simple breakouts (ORB, VWAP reclaim, Squeeze)
- Picks candidate option contract from snapshot and scores them
- Serves `/top5` JSON and a tiny dashboard at `/`

## Setup (non-technical)
1. Unzip the folder
2. Copy `.env.example` to `.env` and add your `POLYGON_API_KEY`
3. Edit `config.yaml` to taste (scan interval, preferred DTE, thresholds)
4. Run (mac/linux):
   ```bash
   ./start.sh
   ```
   Or on Windows, double-click `start.bat`

5. Open http://127.0.0.1:8000 in your browser. The page polls every X seconds and shows the top-5 candidates.

## Notes & Limitations
- This is a starter, not production. It simplifies options snapshot parsing — real Polygon responses are nested and large. You may need to adapt `get_options_snapshot` and `pick_option_from_snapshot` to your Polygon subscription level.
- Test in paper/trade simulator before risking capital.
- Add error handling, persistence, better option-matching rules, and broker integration for auto-execution.

If you want, I can now:
- Adapt the option selection for your exact Polygon subscription (if you tell me your subscription plan).
- Add broker execution integration (IB / Tradier / TDA) for 1-click sends to your account.
- Expand the front-end into a React app with filters, real-time websockets, and trade history.
