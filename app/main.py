\
import os
import asyncio
import math
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks, Request, Form
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import httpx
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yaml

load_dotenv()
API_KEY = os.getenv("POLYGON_API_KEY", "")
if not API_KEY or "YOUR_POLYGON_KEY" in API_KEY:
    print("WARNING: POLYGON_API_KEY not set. The app will not fetch real data until you add your key to .env.")

with open("config.yaml","r") as f:
    cfg = yaml.safe_load(f)

SCAN_INTERVAL = int(os.getenv("POLL_INTERVAL_SECONDS", cfg.get("scan_interval_seconds", 10)))
WATCHLIST = [s.strip().upper() for s in os.getenv("DEFAULT_WATCHLIST","").split(",") if s.strip()][:cfg.get("max_watchlist_size",10)]

app = FastAPI(title="Options Breakout Top-5 Scanner")
app.mount("/static", StaticFiles(directory="static"), name="static")

# In-memory state (simple)
state = {
    "watchlist": WATCHLIST.copy(),
    "last_scan": None,
    "candidates": [],
    "running": False,
    "errors": []
}

# ------------ Helper functions to talk to Polygon --------------
BASE = "https://api.polygon.io"

async def fetch_json(client: httpx.AsyncClient, url: str, params: dict):
    params = params or {}
    params["apiKey"] = API_KEY
    try:
        r = await client.get(url, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        state["errors"].append(f"{datetime.now()}: fetch error {url} {str(e)}")
        return None

async def get_1min_bars(client: httpx.AsyncClient, ticker: str, limit: int = 200):
    url = f"{BASE}/v2/aggs/ticker/{ticker}/range/1/minute/now/now"
    params = {"limit": limit, "adjusted": "false"}
    data = await fetch_json(client, url, params)
    if not data or "results" not in data:
        return None
    df = pd.DataFrame(data["results"])
    if df.empty:
        return None
    df['t'] = pd.to_datetime(df['t'], unit='ms', utc=True)
    df.rename(columns={'o':'open','h':'high','l':'low','c':'close','v':'volume'}, inplace=True)
    df.set_index('t', inplace=True)
    return df

async def get_prev_close(client: httpx.AsyncClient, ticker: str):
    url = f"{BASE}/v2/aggs/ticker/{ticker}/prev"
    data = await fetch_json(client, url, {})
    if not data or "results" not in data:
        return None
    return data["results"][0]

# Note: Polygon has an options snapshot endpoint; we use v3 snapshot/options/{underlying} if available.
async def get_options_snapshot(client: httpx.AsyncClient, underlying: str):
    url = f"{BASE}/v3/snapshot/options/{underlying}"
    data = await fetch_json(client, url, {})
    if not data:
        return None
    return data

# --------------- Simple technical helpers ------------------------
def vwap_from_df(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    typical = (df['high'] + df['low'] + df['close']) / 3.0
    vwap = (typical * df['volume']).cumsum() / df['volume'].cumsum()
    return vwap.iloc[-1]

def compute_rvol(df: pd.DataFrame, window=30):
    if df is None or len(df) < window*2:
        return 1.0
    recent = df['volume'].iloc[-window:].sum()
    prior = df['volume'].iloc[-(window*2):-window].sum()
    if prior <= 0: return 1.0
    return max(0.1, recent / prior)

def opening_range(df: pd.DataFrame, minutes=5):
    if df is None or df.empty:
        return None, None
    first = df.iloc[:minutes]
    return float(first['high'].max()), float(first['low'].min())

# ----------------- Setup detectors ---------------------------------
def detect_orb(df_5m):
    if df_5m is None or len(df_5m) < 6:
        return None
    orh, orl = opening_range(df_5m, minutes=5)
    last_close = float(df_5m['close'].iloc[-1])
    if orh is not None and last_close > orh:
        return {"setup": "ORB_LONG", "orh": orh, "orl": orl}
    if orl is not None and last_close < orl:
        return {"setup": "ORB_SHORT", "orh": orh, "orl": orl}
    return None

def detect_vwap_reclaim(df_1m):
    if df_1m is None or len(df_1m) < 20:
        return None
    vwap = vwap_from_df(df_1m)
    last = df_1m['close'].iloc[-1]
    earlier = df_1m['close'].iloc[-30:-10] if len(df_1m) > 40 else df_1m['close'].iloc[:-1]
    if earlier.empty: return None
    if earlier.min() < vwap and last > vwap and last > df_1m['high'].iloc[-10:].max():
        return {"setup": "VWAP_RECLAIM"}
    return None

def detect_squeeze(df_1m):
    if df_1m is None or len(df_1m) < 50:
        return None
    close = df_1m['close']
    ma = close.rolling(20).mean()
    std = close.rolling(20).std()
    bb_width = (ma + 2*std) - (ma - 2*std)
    if bb_width.iloc[-10:-1].mean() < (0.002 * close.iloc[-1]):
        if close.iloc[-1] > df_1m['high'].iloc[-11:-1].max():
            return {"setup": "SQUEEZE_BREAKOUT"}
    return None

# -------------- Option selection & scoring -------------------------
def pick_option_from_snapshot(snapshot: dict, side: str, target_delta: float, cfg_local: dict):
    if not snapshot:
        return None
    results = snapshot.get('results') or snapshot.get('underlying') or snapshot
    cand = None
    if isinstance(results, dict):
        for key in ['options','contracts','chains']:
            if key in results and isinstance(results[key], list):
                if results[key]:
                    cand = results[key][0]
                    break
    elif isinstance(results, list):
        cand = results[0]
    if cand is None:
        return None
    opt = {
        "contract": cand.get('sym') if isinstance(cand, dict) else str(cand),
        "mid": (cand.get('ask',0) + cand.get('bid',0))/2 if isinstance(cand, dict) else None,
        "bid": cand.get('bid') if isinstance(cand, dict) else None,
        "ask": cand.get('ask') if isinstance(cand, dict) else None,
        "oi": cand.get('openInterest', cand.get('oi',0)) if isinstance(cand, dict) else 0,
        "volume": cand.get('volume',0) if isinstance(cand, dict) else 0,
        "delta": cand.get('delta', None) if isinstance(cand, dict) else None,
        "expiry": cand.get('expirationDate', None) if isinstance(cand, dict) else None,
        "strike": cand.get('strikePrice', None) if isinstance(cand, dict) else None,
        "type": side
    }
    return opt

def score_candidate(ticker: str, setup: dict, rvol: float, option: dict):
    s = 0.0
    if setup['setup'].startswith("ORB"):
        s += 40.0
    elif setup['setup'].startswith("VWAP"):
        s += 30.0
    elif setup['setup'].startswith("SQUEEZE"):
        s += 28.0
    s += min(40.0, (rvol - 1.0) * 20.0)
    if option:
        if option.get('bid') and option.get('ask') and option.get('mid'):
            spread = (option['ask'] - option['bid']) / max(0.01, option['mid'])
            if spread < 0.05:
                s += 20.0
            elif spread < 0.1:
                s += 10.0
        if option.get('oi',0) > 500:
            s += 10.0
    return max(0.0, min(100.0, s))

# ----------------- Scanner loop ------------------------------------
async def scan_once(client: httpx.AsyncClient):
    candidates = []
    for ticker in state['watchlist']:
        try:
            df1 = await get_1min_bars(client, ticker, limit=200)
            if df1 is None:
                continue
            rvol = compute_rvol(df1, window=30)
            df5 = df1.resample('5T').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'})
            setup = detect_orb(df5) or detect_vwap_reclaim(df1) or detect_squeeze(df1)
            if not setup:
                continue
            snapshot = await get_options_snapshot(client, ticker)
            side = 'CALL' if setup['setup'].endswith('LONG') or 'LONG' in setup['setup'] or 'RECLAIM' in setup['setup'] or 'SQUEEZE' in setup['setup'] else 'PUT'
            opt = pick_option_from_snapshot(snapshot, side, cfg.get('target_delta',0.35), cfg)
            sc = score_candidate(ticker, setup, rvol, opt)
            cand = {
                "symbol": ticker,
                "setup": setup,
                "rvol": round(float(rvol),2) if rvol else None,
                "option": opt,
                "score": round(sc,1),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            candidates.append(cand)
        except Exception as e:
            state['errors'].append(f"{datetime.now()}: scan error {ticker} {str(e)}")
            continue
    candidates_sorted = sorted(candidates, key=lambda x: x['score'], reverse=True)[:cfg.get('top_k',5)]
    state['candidates'] = candidates_sorted
    state['last_scan'] = datetime.now(timezone.utc).isoformat()

async def background_scanner():
    async with httpx.AsyncClient() as client:
        while True:
            try:
                await scan_once(client)
            except Exception as e:
                state['errors'].append(f"{datetime.now()}: background error {str(e)}")
            await asyncio.sleep(SCAN_INTERVAL)

@app.on_event("startup")
async def startup_event():
    loop = asyncio.get_event_loop()
    loop.create_task(background_scanner())
    state['running'] = True

# ----------------- API endpoints -----------------------------------
class WatchlistIn(BaseModel):
    tickers: List[str]

@app.post("/watchlist")
async def set_watchlist(wl: WatchlistIn):
    wl2 = [s.strip().upper() for s in wl.tickers][:cfg.get('max_watchlist_size',10)]
    state['watchlist'] = wl2
    return {"status":"ok","watchlist":state['watchlist']}

@app.get("/top5")
async def top5():
    return JSONResponse({"top5": state['candidates'], "last_scan': state['last_scan']})

@app.get("/status")
async def status():
    return {"running": state['running'], "watchlist": state['watchlist'], "last_scan": state['last_scan'], "errors": state['errors'][-20:]}

@app.get("/", response_class=HTMLResponse)
async def homepage():
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    return {"status":"ok","time":datetime.now(timezone.utc).isoformat()}
