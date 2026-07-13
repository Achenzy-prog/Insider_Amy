"""
diagnose_stage1.py
Pinpoints why Metric 1's `rows` list came out empty in 02_stage1_filter.py.

Run from project root: python diagnose_stage1.py
"""

import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from config import WALLET, PROC_DIR
from algo.metrics import compute_wallet_roi

print("=== Loading preprocessed data ===")
trades  = pd.read_parquet(PROC_DIR / "trades_clean.parquet")
peers   = pd.read_parquet(PROC_DIR / "peer_stats.parquet")

print(f"trades_clean.parquet: {len(trades)} rows")
print(f"  columns: {list(trades.columns)}")
print(f"peer_stats.parquet:   {len(peers)} rows")
print(f"  columns: {list(peers.columns)}")

print("\n=== Checking wallet match ===")
print(f"Target wallet: {WALLET}")
print(f"Sample wallets in trades_clean: {trades['wallet'].head(5).tolist()}")
matches = trades[trades["wallet"].str.lower() == WALLET.lower()]
print(f"Rows matching target wallet (case-insensitive): {len(matches)}")

print("\n=== compute_wallet_roi output ===")
wallet_rois = compute_wallet_roi(trades, WALLET)
print(f"Number of markets with computed ROI: {len(wallet_rois)}")
if wallet_rois:
    print(f"Sample: {dict(list(wallet_rois.items())[:3])}")
else:
    print("  ⚠ EMPTY. This means either:")
    print("    (a) no trades matched the wallet (check the match count above), or")
    print("    (b) every matched market had buys==0 (so ROI is undefined)")

print("\n=== peer_stats coverage ===")
if len(peers) == 0:
    print("  ⚠ peer_stats.parquet is EMPTY.")
    print("  This means 01_preprocess.py's peer-stats loop produced nothing.")
    print("  Likely cause: market_all_trades.json wallet column wasn't detected.")
else:
    cids_with_roi = set(wallet_rois.keys())
    cids_in_peers = set(peers["condition_id"])
    overlap = cids_with_roi & cids_in_peers
    print(f"  Markets with wallet ROI: {len(cids_with_roi)}")
    print(f"  Markets in peer_stats:   {len(cids_in_peers)}")
    print(f"  Overlap (this is what Metric 1 actually uses): {len(overlap)}")
    if len(overlap) == 0 and len(cids_with_roi) > 0:
        print("  ⚠ ZERO OVERLAP — condition_ids don't match between the two tables.")
        print(f"    Sample wallet ROI cid: {list(cids_with_roi)[0] if cids_with_roi else 'N/A'}")
        print(f"    Sample peer_stats cid: {list(cids_in_peers)[0] if cids_in_peers else 'N/A'}")