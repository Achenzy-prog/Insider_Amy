"""
experiment_ablation_9wallets.py
================================
Feature ablation study on the original 9 unaugmented wallets.

No augmentation. Each wallet is one data point. Leave-One-Out (LOO) CV
leaves one wallet out at a time — 9 folds, each training on 8 wallets
and testing on 1.

Two feature sets tested in parallel:

A) Stage features (used in the main ordinal model):
   buy_fraction, mean_trade_size_usd, mean_size_vs_daily,
   mean_dt_hours, n_markets, mean_price_paid

B) Math pipeline features:
   adj_roi, wilson_lb, hhi, mean_dt_hours, age_flag

For each feature set, forward stepwise selection reveals which feature
contributes most at each step.

Note on evaluation with 9 wallets:
  With only 9 data points and 3 insider, precision/recall/F1 are very
  coarse — each wallet is either detected or not. We report:
  - Insider recall: how many of the 3 insiders are ranked in top 3?
  - Ranking: mean rank of the 3 insider wallets (lower = better)
  - NDCG: ranking quality across all 9 wallets

Run: python experiment_ablation_9wallets.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
import warnings
warnings.filterwarnings("ignore")

from model import OrdinalLogisticRegression

OUT_DIR = Path("data/experiment_ablation_9wallets")
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_MAP = {"Average": 0, "Skilled": 1, "Insider": 2}
LABEL_NAMES = ["Average", "Skilled", "Insider"]
L2 = 1.0

WALLETS = {
    '0x1f742b63dFDc4b6d58890cbBC380b125D1C6558D': ('OTW10T',       'Average'),
    '0x30F91B6dF5a62aB2e504045E25fD449a23AFfBfC': ('taerv534',     'Average'),
    '0x094e3e7c5486ca7f5EFD45De09857B873c70F10b': ('Minecraft',    'Average'),
    '0xEe50A31C3f5A7C77824b12a941A54388A2827Ed6': ('AlphaRaccoon', 'Insider'),
    '0xc0a005750061E6B5c5699c395A09FD238116aC10': ('OmerZiv',      'Insider'),
    '0x31a56e9E690c621eD21De08Cb559e9524Cdb8eD9': ('Venezuela',    'Insider'),
    '0xcE41369f5Db6F155Fb6D2AE31F6D57052904Fa02': ('Gambler2026',  'Skilled'),
    '0xe7387473B067235436884d16799777cF279EdF65': ('huludubu',     'Skilled'),
    '0xc851CD9BeE7D262AfD78674F861f9F576A12CD2a': ('Betwick',      'Skilled'),
}


# ── Load raw data and compute all features ────────────────────────────────────

print("Loading raw data and computing features for 9 wallets...")
trades  = pd.read_parquet('data/external/trades.parquet')
markets = pd.read_parquet('data/external/markets.parquet')
trades['timestamp'] = pd.to_datetime(trades['timestamp'], utc=True)
trades['next_price_jump_timestamp'] = pd.to_datetime(
    trades['next_price_jump_timestamp'], utc=True, errors='coerce'
)

# Merge market info onto trades
trades = trades.merge(
    markets[['market_id','event_id','event_title',
             'token1_final_price','token2_final_price',
             'token1_id','token2_id']],
    on='market_id', how='left'
)

rows = []
for addr, (name, label) in WALLETS.items():
    wdf = trades[trades['wallet_address'].str.lower() == addr.lower()].copy()

    # ── Stage features ────────────────────────────────────────────────────────
    buy_fraction        = float((wdf['direction'].str.upper()=='BUY').mean())
    mean_trade_size_usd = float(wdf['usd_amount'].mean())

    valid_vol = wdf['rolling_24h_volume_usd'].replace(0, np.nan).dropna()
    mean_size_vs_daily = float(
        (wdf.loc[valid_vol.index,'usd_amount'] / valid_vol).mean()
    ) if len(valid_vol) > 0 else np.nan

    has_jump = wdf['next_price_jump_timestamp'].notna()
    if has_jump.sum() > 0:
        dt = (wdf.loc[has_jump,'timestamp'] -
              wdf.loc[has_jump,'next_price_jump_timestamp']
              ).dt.total_seconds() / 3600
        mean_dt_hours = float(np.clip(dt.mean(), -72, 72))
    else:
        mean_dt_hours = np.nan

    n_markets       = float(wdf['market_id'].nunique())
    mean_price_paid = float(wdf['price'].mean())

    # ── Math pipeline features ────────────────────────────────────────────────

    # adj_roi: mean z-score of wallet's entry vs market distribution
    wdf['trade_roi'] = np.where(
        wdf['direction'].str.upper() == 'BUY',
        (wdf['price'] - wdf['market_mean_entry_price']) /
            wdf['market_mean_entry_price'].replace(0, np.nan),
        np.nan
    )
    wdf['roi_z'] = (wdf['trade_roi'] - wdf['market_mean_roi']) / \
                    wdf['market_std_roi'].replace(0, np.nan)
    adj_roi = float(wdf['roi_z'].mean()) if wdf['roi_z'].notna().any() else np.nan

    # wilson_lb: Wilson score lower bound on win rate
    buys = wdf[wdf['direction'].str.upper() == 'BUY'].copy()
    if len(buys) > 0:
        buys['won'] = np.where(
            buys['outcome'].str.upper() == 'YES',
            (buys['token1_final_price'] == 1.0).astype(float),
            (buys['token2_final_price'] == 1.0).astype(float)
        )
        n = len(buys); w = float(buys['won'].sum()); z = 1.96
        wilson_lb = float(
            (w/n + z**2/(2*n) - z*np.sqrt(w*(n-w)/n**3 + z**2/(4*n**2))) /
            (1 + z**2/n)
        ) if n > 0 else np.nan
    else:
        wilson_lb = np.nan

    # hhi: Herfindahl-Hirschman Index by event (volume concentration)
    vol_by_event = wdf.groupby('event_id')['usd_amount'].sum()
    total_vol    = vol_by_event.sum()
    hhi = float(((vol_by_event/total_vol)**2).sum()) if total_vol > 0 else np.nan

    # mean_dt_hours already computed above (shared between both feature sets)

    # age_flag: 1 if account active < 30 days
    age_days = (wdf['timestamp'].max() - wdf['timestamp'].min()).days
    age_flag = float(1 if age_days < 30 else 0)

    rows.append({
        'name': name, 'label': label, 'address': addr,
        # Stage features
        'buy_fraction':        buy_fraction,
        'mean_trade_size_usd': mean_trade_size_usd,
        'mean_size_vs_daily':  mean_size_vs_daily,
        'mean_dt_hours':       mean_dt_hours,
        'n_markets':           n_markets,
        'mean_price_paid':     mean_price_paid,
        # Math pipeline features
        'adj_roi':             adj_roi,
        'wilson_lb':           wilson_lb,
        'hhi':                 hhi,
        'age_flag':            age_flag,
    })

df = pd.DataFrame(rows)
df['y'] = df['label'].map(LABEL_MAP).astype(int)
df['y_bin'] = (df['y'] == 2).astype(int)

print(f"  9 wallets loaded")
print(f"\nFeature values per wallet:")
all_feats = ['buy_fraction','mean_trade_size_usd','mean_size_vs_daily',
             'mean_dt_hours','n_markets','mean_price_paid',
             'adj_roi','wilson_lb','hhi','age_flag']
print(df.set_index('name')[['label']+all_feats].round(4).to_string())


# ── LOO evaluation (9 folds) ──────────────────────────────────────────────────

loo = LeaveOneOut()

def evaluate_9(feature_subset, df):
    """
    Leave-One-Out CV on the 9 raw wallets.
    With only 9 points, we report ranking metrics rather than precision/recall.
    """
    X   = df[feature_subset].values.astype(np.float64)
    y   = df['y'].values.astype(int)
    yb  = df['y_bin'].values.astype(int)
    names = df['name'].values
    labels = df['label'].values

    preds = np.zeros(len(df))

    for tr, te in loo.split(X):
        Xtr, Xte = X[tr], X[te]
        ytr       = y[tr]
        sc        = StandardScaler()
        Xtr_s     = sc.fit_transform(Xtr)
        Xte_s     = sc.transform(Xte)
        m = OrdinalLogisticRegression(l2=L2)
        m.fit(Xtr_s, ytr)
        preds[te] = m.p_insider(Xte_s)

    # Rank by P(Insider) descending
    ranked = pd.DataFrame({
        'name': names, 'label': labels, 'y_bin': yb, 'p_insider': preds
    }).sort_values('p_insider', ascending=False).reset_index(drop=True)
    ranked['rank'] = ranked.index + 1

    insider_ranks = ranked[ranked['y_bin']==1]['rank'].tolist()
    insider_ps    = ranked[ranked['y_bin']==1]['p_insider'].tolist()
    mean_rank     = float(np.mean(insider_ranks))
    insiders_in_top3 = int(sum(r<=3 for r in insider_ranks))

    # NDCG
    dcg  = sum(ranked['y_bin'].iloc[i]/np.log2(i+2) for i in range(len(ranked)))
    idcg = sum(1.0/np.log2(i+2) for i in range(yb.sum()))
    ndcg = dcg/idcg if idcg > 0 else 0.0

    return {
        'mean_rank':        round(mean_rank, 2),
        'insiders_in_top3': insiders_in_top3,
        'ndcg':             round(ndcg, 3),
        'insider_ranks':    insider_ranks,
        'insider_ps':       [round(p,4) for p in insider_ps],
        'ranking':          ranked,
    }


# ── Forward stepwise: Stage features ─────────────────────────────────────────

STAGE_FEATURES = ['buy_fraction','mean_trade_size_usd','mean_size_vs_daily',
                  'mean_dt_hours','n_markets','mean_price_paid']
# Drop degenerate
STAGE_FEATURES = [f for f in STAGE_FEATURES if df[f].nunique() > 1]

print(f"\n\n{'='*70}")
print("PART A — FORWARD STEPWISE: STAGE FEATURES (on 9 raw wallets)")
print("="*70)
print("Features:", STAGE_FEATURES)
print()

selected=[]; remaining=STAGE_FEATURES.copy(); stage_rows=[]
for step in range(len(STAGE_FEATURES)):
    print(f"── Step {step+1}: current = {selected if selected else '(empty)'} ──")
    print(f"  {'Feature':25s}  {'NDCG':>6}  {'Mean Rank':>9}  {'Insiders in Top-3':>18}  "
          f"{'Insider Ranks'}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*9}  {'-'*18}  {'-'*20}")

    cands = []
    for cand in remaining:
        feat_df = df[selected+[cand]+['y','y_bin','name','label']].dropna()
        if len(feat_df) < 9:
            print(f"  {cand:25s}  (insufficient data after dropna)")
            continue
        m = evaluate_9(selected+[cand], feat_df)
        cands.append((cand, m))
        print(f"  {cand:25s}  {m['ndcg']:>6.3f}  {m['mean_rank']:>9.2f}  "
              f"{m['insiders_in_top3']:>18}  {m['insider_ranks']}")

    if not cands:
        break
    best_feat, best_m = max(cands, key=lambda x: (x[1]['ndcg'], -x[1]['mean_rank']))
    selected.append(best_feat); remaining.remove(best_feat)
    print(f"\n  → Selected: {best_feat}  "
          f"NDCG={best_m['ndcg']:.3f}  MeanRank={best_m['mean_rank']:.2f}\n")
    stage_rows.append({
        'step': step+1, 'feature_added': best_feat,
        'features_so_far': ", ".join(selected),
        'ndcg': best_m['ndcg'],
        'mean_rank': best_m['mean_rank'],
        'insiders_in_top3': best_m['insiders_in_top3'],
        'insider_ranks': str(best_m['insider_ranks']),
    })

print("STAGE FEATURE SUMMARY:")
prev_ndcg=0.0
for r in stage_rows:
    d=r['ndcg']-prev_ndcg
    arrow="▲" if d>0 else("▼" if d<0 else "─")
    print(f"  {arrow} Step {r['step']}: +{r['feature_added']:25s}  "
          f"NDCG {'+' if d>=0 else ''}{d:.3f}  (cumulative={r['ndcg']:.3f})  "
          f"InsidersTop3={r['insiders_in_top3']}/3  MeanRank={r['mean_rank']}")
    prev_ndcg=r['ndcg']

pd.DataFrame(stage_rows).to_csv(OUT_DIR/"stage_stepwise.csv", index=False)


# ── Forward stepwise: Math pipeline features ──────────────────────────────────

MATH_FEATURES = ['adj_roi','wilson_lb','hhi','mean_dt_hours','age_flag']
MATH_FEATURES = [f for f in MATH_FEATURES if df[f].nunique() > 1]

print(f"\n\n{'='*70}")
print("PART B — FORWARD STEPWISE: MATH PIPELINE FEATURES (on 9 raw wallets)")
print("="*70)
print("Features:", MATH_FEATURES)
print()

selected=[]; remaining=MATH_FEATURES.copy(); math_rows=[]
for step in range(len(MATH_FEATURES)):
    print(f"── Step {step+1}: current = {selected if selected else '(empty)'} ──")
    print(f"  {'Feature':25s}  {'NDCG':>6}  {'Mean Rank':>9}  {'Insiders in Top-3':>18}  "
          f"{'Insider Ranks'}")
    print(f"  {'-'*25}  {'-'*6}  {'-'*9}  {'-'*18}  {'-'*20}")

    cands = []
    for cand in remaining:
        feat_df = df[selected+[cand]+['y','y_bin','name','label']].dropna()
        if len(feat_df) < 9:
            print(f"  {cand:25s}  (insufficient data after dropna)")
            continue
        m = evaluate_9(selected+[cand], feat_df)
        cands.append((cand, m))
        print(f"  {cand:25s}  {m['ndcg']:>6.3f}  {m['mean_rank']:>9.2f}  "
              f"{m['insiders_in_top3']:>18}  {m['insider_ranks']}")

    if not cands:
        break
    best_feat, best_m = max(cands, key=lambda x: (x[1]['ndcg'], -x[1]['mean_rank']))
    selected.append(best_feat); remaining.remove(best_feat)
    print(f"\n  → Selected: {best_feat}  "
          f"NDCG={best_m['ndcg']:.3f}  MeanRank={best_m['mean_rank']:.2f}\n")
    math_rows.append({
        'step': step+1, 'feature_added': best_feat,
        'features_so_far': ", ".join(selected),
        'ndcg': best_m['ndcg'],
        'mean_rank': best_m['mean_rank'],
        'insiders_in_top3': best_m['insiders_in_top3'],
        'insider_ranks': str(best_m['insider_ranks']),
    })

print("MATH PIPELINE SUMMARY:")
prev_ndcg=0.0
for r in math_rows:
    d=r['ndcg']-prev_ndcg
    arrow="▲" if d>0 else("▼" if d<0 else "─")
    print(f"  {arrow} Step {r['step']}: +{r['feature_added']:25s}  "
          f"NDCG {'+' if d>=0 else ''}{d:.3f}  (cumulative={r['ndcg']:.3f})  "
          f"InsidersTop3={r['insiders_in_top3']}/3  MeanRank={r['mean_rank']}")
    prev_ndcg=r['ndcg']

pd.DataFrame(math_rows).to_csv(OUT_DIR/"math_stepwise.csv", index=False)


# ── Final full ranking under best feature set ──────────────────────────────────

print(f"\n\n{'='*70}")
print("FINAL FULL RANKINGS — All 9 wallets")
print("="*70)

for label, feats, saved_rows in [
    ("Stage features (all)", STAGE_FEATURES, stage_rows),
    ("Math pipeline (all)",  MATH_FEATURES,  math_rows),
]:
    feat_df = df[feats+['y','y_bin','name','label']].dropna()
    m = evaluate_9(feats, feat_df)
    print(f"\n── {label} ──")
    print(f"   NDCG={m['ndcg']:.3f}  MeanInsiderRank={m['mean_rank']:.2f}  "
          f"InsidersTop3={m['insiders_in_top3']}/3")
    print()
    print(f"  {'Rank':>4}  {'Name':15s}  {'Label':8s}  {'P(Insider)':>10}  Correct?")
    print(f"  {'----':>4}  {'-'*15}  {'-------':8s}  {'----------':>10}  --------")
    for _, row in m['ranking'].iterrows():
        marker = "✓ INSIDER" if row['label']=="Insider" else ""
        print(f"  {int(row['rank']):>4}  {row['name']:15s}  {row['label']:8s}  "
              f"{row['p_insider']:>10.4f}  {marker}")

print(f"\n✓ Saved to {OUT_DIR}/")
print("  stage_stepwise.csv — stage feature ablation results")
print("  math_stepwise.csv  — math pipeline ablation results")