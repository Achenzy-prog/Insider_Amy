"""
check_resolution_fields.py
Before rewriting win-labelling AGAIN, check which fields in markets.json
actually carry reliable resolution signal: closed, umaResolutionStatuses,
or the tail of price_histories.json candles.

Run from project root: python check_resolution_fields.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import RAW_DIR

with open(RAW_DIR / "markets.json") as f:
    markets = json.load(f)

print(f"Total markets: {len(markets)}\n")

print("=== closed / active / outcomePrices for first 15 markets ===")
for m in markets[:15]:
    print(f"  cid={m.get('conditionId', '')[:16]}...  "
          f"closed={m.get('closed')}  active={m.get('active')}  "
          f"outcomePrices={m.get('outcomePrices')}  "
          f"umaStatuses={m.get('umaResolutionStatuses')}")

n_closed_true  = sum(1 for m in markets if m.get("closed") is True)
n_closed_false = sum(1 for m in markets if m.get("closed") is False)
n_active_true  = sum(1 for m in markets if m.get("active") is True)
n_active_false = sum(1 for m in markets if m.get("active") is False)

print(f"\nclosed=True:  {n_closed_true}")
print(f"closed=False: {n_closed_false}")
print(f"active=True:  {n_active_true}")
print(f"active=False: {n_active_false}")

print("\n=== outcomePrices distribution among closed=True markets ===")
for m in markets:
    if m.get("closed") is True:
        print(f"  cid={m.get('conditionId', '')[:16]}...  outcomePrices={m.get('outcomePrices')}")


print("\n=== price_histories.json check (last candle per market) ===")
with open(RAW_DIR / "price_histories.json") as f:
    ph = json.load(f)

print(f"Markets with price history: {len(ph)}")
shown = 0
for cid, history in ph.items():
    if not history:
        continue
    last_candle = sorted(history, key=lambda x: x.get("t", 0))[-1]
    print(f"  cid={cid[:16]}...  last candle price={last_candle.get('p')}  "
          f"at t={last_candle.get('t')}")
    shown += 1
    if shown >= 15:
        break