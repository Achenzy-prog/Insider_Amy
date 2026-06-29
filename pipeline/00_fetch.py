"""
pipeline/00_fetch.py
====================
Pulls all raw data needed for Stages 1–4 from the public Polymarket APIs.
No authentication required.

Parallelized with a thread pool: the original sequential version made
one HTTP request at a time with a sleep() between each, which is the
correct way to be polite to an API but is brutally slow across ~1000+
markets. Network I/O like this is the textbook case for threads (the
GIL doesn't matter here since we're waiting on sockets, not doing CPU work).

Outputs (all JSON, written to data/raw/):
  trades.json              — every trade by the target wallet
  market_ids.json          — unique conditionIds from those trades
  markets.json             — Gamma metadata for each market
  price_histories.json     — 1-min CLOB candles for each market
  market_all_trades.json   — all trades by all wallets on those markets
  order_books.json         — current order book snapshot per market
  profile.json             — wallet profile (account creation date)
  closed_positions.json    — resolved positions for win/loss labels
  clob_trades.json         — CLOB-level trade records for the target wallet
"""

import sys, json, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    WALLET, DATA_API, GAMMA_API, CLOB_API,
    RAW_DIR, WINDOW_START_TS, WINDOW_END_TS,
)

# Tune this based on how aggressively Polymarket rate-limits you.
# 10-20 is usually safe for public APIs without auth; lower it if you
# start seeing lots of 429s in the output.
MAX_WORKERS = 16


# ── Helpers ───────────────────────────────────────────────────────────────────

def get(url, params=None, retries=4):
    """GET with exponential backoff. Thread-safe — each call uses its own request."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 2 ** (attempt + 2)
                time.sleep(wait)
            else:
                break
        except Exception:
            wait = 2 ** attempt
            time.sleep(wait)
    return None


def save(obj, name):
    path = RAW_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(obj, f)
    n = len(obj) if isinstance(obj, (list, dict)) else "—"
    print(f"  ✓ {name}.json  ({n} records)")
    return path


def paginate(url, params, page_size=500):
    """Collect all pages from a paginated endpoint. Sequential — pages depend
    on each other (offset), so this part can't be parallelized per-call."""
    results = []
    offset = 0
    while True:
        batch = get(url, {**params, "limit": page_size, "offset": offset})
        if not batch:
            break
        results.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size
    return results


def fetch_many(items, fetch_fn, desc, max_workers=MAX_WORKERS):
    """
    Run fetch_fn(item) concurrently across `items` using a thread pool.
    fetch_fn must return (key, value) or None to skip.
    Returns a dict {key: value} for all successful fetches.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_fn, item): item for item in items}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"  {desc}"):
            try:
                outcome = future.result()
            except Exception:
                outcome = None
            if outcome is not None:
                key, value = outcome
                results[key] = value
    return results


# ── Step 1: All trades by the target wallet ───────────────────────────────────
# Sequential (pagination is inherently sequential — each page needs the offset
# from the previous one). This step is fast anyway since it's just one wallet.

print(f"\n[1/8] Fetching all trades for {WALLET[:10]}...")
trades = paginate(f"{DATA_API}/trades", {
    "user": WALLET,
    "takerOnly": "false",
})
save(trades, "trades")

market_ids = list({t["conditionId"] for t in trades if t.get("conditionId")})
save(market_ids, "market_ids")
print(f"      {len(market_ids)} unique markets found")


# ── Step 2: Gamma market metadata (parallelized) ──────────────────────────────

print(f"\n[2/8] Fetching market metadata ({len(market_ids)} markets, {MAX_WORKERS} workers)...")

# IMPORTANT: gamma-api.polymarket.com/markets?condition_id=X is NOT a
# documented or reliable filter — in practice it silently ignores the
# parameter and returns the same default market for every request,
# regardless of which condition_id was passed. This caused 00_fetch.py to
# write 59 duplicate copies of one market into markets.json.
#
# The reliable approach is to query the CLOB API directly by condition_id
# via its dedicated endpoint, since that one DOES filter correctly (it's
# how order books and market microstructure are looked up on-chain).
# We fall back to Gamma's slug-based lookup only if needed.

def fetch_market(cid):
    # Primary: CLOB API's /markets/{condition_id} path endpoint reliably
    # filters by condition_id (this is how trading clients fetch market info).
    data = get(f"{CLOB_API}/markets/{cid}")
    if data and (data.get("condition_id") == cid or data.get("conditionId") == cid):
        return (cid, data)

    # Fallback: try Gamma's path-based single-market lookup (some Gamma
    # deployments key this by conditionId path segment rather than slug).
    data = get(f"{GAMMA_API}/markets/{cid}")
    if data and (data.get("conditionId") == cid or data.get("condition_id") == cid):
        return (cid, data)

    return None

market_map = fetch_many(market_ids, fetch_market, "markets")
markets = list(market_map.values())

# Sanity check: if many markets came back with the same conditionId, the API
# is not filtering correctly (this exact bug happened once already — see git
# history). Catch it here instead of letting it silently corrupt every
# downstream stage.
returned_cids = [m.get("conditionId") or m.get("condition_id") for m in markets]
unique_cids = set(returned_cids)
if len(markets) > 5 and len(unique_cids) < len(markets) * 0.5:
    print(f"  ⚠ WARNING: {len(markets)} markets fetched but only {len(unique_cids)} "
          f"unique conditionIds returned. The API may be ignoring the lookup "
          f"parameter and returning duplicate/default records. Inspect markets.json "
          f"before trusting downstream results.")

save(markets, "markets")


# ── Step 3: Price histories (parallelized) ────────────────────────────────────

print(f"\n[3/8] Fetching price histories ({MAX_WORKERS} workers)...")

# Build conditionId → YES token_id map.
# Markets now come from the CLOB API (see fetch_market above), which returns
# a clean `tokens: [{outcome, price, token_id, winner}, ...]` array directly —
# no JSON-string parsing needed. We keep the old Gamma-style `clobTokenIds`
# string fallback in case any record came from the Gamma fallback path instead.
market_tokens = {}
market_resolution = {}  # cid -> {token_id: winner_bool}, used later for win/loss

for m in markets:
    cid = m.get("conditionId") or m.get("condition_id")
    if not cid:
        continue

    tokens = m.get("tokens")  # CLOB API shape
    if tokens:
        # tokens[i] = {"outcome": ..., "price": ..., "token_id": ..., "winner": bool}
        yes_token = next((t for t in tokens if t.get("outcome", "").lower() == "yes"), tokens[0])
        market_tokens[cid] = yes_token.get("token_id")
        market_resolution[cid] = {t.get("token_id"): t.get("winner", False) for t in tokens}
        continue

    # Fallback: Gamma-style clobTokenIds as a JSON-encoded string
    raw_tokens = m.get("clobTokenIds")
    if not cid or not raw_tokens:
        continue
    try:
        tokens = json.loads(raw_tokens) if isinstance(raw_tokens, str) else raw_tokens
    except (json.JSONDecodeError, TypeError):
        continue
    if tokens and len(tokens) > 0:
        market_tokens[cid] = tokens[0]

def fetch_price_history(item):
    cid, token_id = item
    data = get(f"{CLOB_API}/prices-history", params={
        "market": token_id,
        "fidelity": 1,
        "startTs": WINDOW_START_TS,
        "endTs":   WINDOW_END_TS,
    })
    if data and data.get("history"):
        return (cid, data["history"])
    return None

price_histories = fetch_many(list(market_tokens.items()), fetch_price_history, "price histories")
save(price_histories, "price_histories")

# Save the resolution map separately — this is the reliable win/loss ground
# truth (token.winner from the CLOB API), far more trustworthy than trying to
# infer resolution from outcomePrices, which often just reflects a live/
# unresolved price snapshot rather than final settlement.
save(market_resolution, "market_resolution")


# ── Step 4: All traders on AlphaRaccoon's markets (parallelized) ─────────────
# Each market's trades are still paginated sequentially within fetch_fn
# (pagination is inherently sequential), but markets run in parallel with
# each other.

print(f"\n[4/8] Fetching all trades per market ({MAX_WORKERS} workers)...")

def fetch_market_trades(cid):
    data = paginate(f"{DATA_API}/trades", {
        "market": cid,
        "takerOnly": "false",
    }, page_size=500)
    if data:
        return (cid, data)
    return None

market_all_trades = fetch_many(market_ids, fetch_market_trades, "market trades")
save(market_all_trades, "market_all_trades")


# ── Step 5: Order book snapshots (parallelized) ───────────────────────────────

print(f"\n[5/8] Fetching order books ({MAX_WORKERS} workers)...")

def fetch_order_book(item):
    cid, token_id = item
    data = get(f"{CLOB_API}/book", params={"token_id": token_id})
    if data:
        return (cid, data)
    return None

order_books = fetch_many(list(market_tokens.items()), fetch_order_book, "order books")
save(order_books, "order_books")


# ── Step 6: Wallet profile ────────────────────────────────────────────────────
# Single request, nothing to parallelize.

print(f"\n[6/8] Fetching wallet profile...")
profile = get(f"{DATA_API}/profile", params={"address": WALLET})
if not profile:
    profile = get(f"{DATA_API}/profiles/{WALLET}")
save(profile or {}, "profile")


# ── Step 7: Closed positions ──────────────────────────────────────────────────
# Pagination is sequential by nature, single wallet — fast regardless.

print(f"\n[7/8] Fetching closed positions...")
closed = paginate(f"{DATA_API}/positions", {
    "user": WALLET,
    "sizeThreshold": 0,
})
save(closed or [], "closed_positions")


# ── Step 8: CLOB trade-level data (parallelized) ──────────────────────────────

print(f"\n[8/8] Fetching CLOB-level trade records ({MAX_WORKERS} workers)...")

def fetch_clob_trades(cid):
    data = get(f"{CLOB_API}/trades", params={
        "market": cid,
        "maker_address": WALLET,
        "limit": 500,
    })
    if data and isinstance(data, list):
        return (cid, data)
    return None

clob_trades_map = fetch_many(market_ids, fetch_clob_trades, "CLOB trades")
clob_trades = [trade for trades_list in clob_trades_map.values() for trade in trades_list]
save(clob_trades, "clob_trades")


print("\n=== Fetch complete ===")
print(f"All raw files written to: {RAW_DIR}")