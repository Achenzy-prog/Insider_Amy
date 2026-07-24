"""
experiment_27fold.py
====================
27-fold cross-validation: all combinations of (1 insider, 1 skilled, 1 average)
held out as test set.

With 3 insider × 3 skilled × 3 average wallets, there are 3×3×3 = 27
possible test sets of size 300 (100 copies each).

For each fold:
  Test:  1 insider + 1 skilled + 1 average (300 wallets)
  Train: remaining 6 source wallets (600 wallets)

This exhaustively tests every possible grouping and gives a stable
estimate of mean train/test performance that isn't sensitive to any
single data split choice.

Run: python experiment_27fold.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from itertools import product
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from model import OrdinalLogisticRegression

DATA_PATH = Path("data/external/9_wallet_stage_features.parquet")
OUT_DIR   = Path("data/experiment_27fold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    print(f"⚠  {DATA_PATH} not found.")
    sys.exit(1)

FEATURES  = ['buy_fraction','mean_trade_size_usd','mean_size_vs_daily',
             'mean_dt_hours','n_markets','mean_price_paid']
LABEL_MAP = {"Average": 0, "Skilled": 1, "Insider": 2}
L2        = 1.0

INSIDERS = ["AlphaRaccoon", "OmerZiv", "Venezuela"]
SKILLED  = ["Gambler2026",  "huludubu", "Betwick"]
AVERAGE  = ["OTW10T",       "taerv534", "Minecraft"]


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading 900-wallet augmented dataset...")
raw = pd.read_parquet(DATA_PATH)
raw = raw.rename(columns={
    "buy_sell_ratio":         "buy_fraction",
    "mean_size_vs_daily_vol": "mean_size_vs_daily",
    "num_markets":            "n_markets",
})
raw["y"]     = raw["label_category"].map(LABEL_MAP)
raw["y_bin"] = (raw["label_category"] == "Insider").astype(int)
for f in FEATURES:
    raw[f] = raw[f].astype(float)

df = raw[FEATURES + ["y","y_bin","source_wallet_name",
                     "label_category"]].dropna().reset_index(drop=True)
print(f"  {len(df)} rows  |  {df['label_category'].value_counts().to_dict()}\n")


# ── 27-fold CV ────────────────────────────────────────────────────────────────

print("=" * 70)
print("27-FOLD CROSS-VALIDATION")
print("=" * 70)
print(f"\nAll {len(INSIDERS)}×{len(SKILLED)}×{len(AVERAGE)} = "
      f"{len(INSIDERS)*len(SKILLED)*len(AVERAGE)} combinations of "
      f"(insider, skilled, average) test sets.\n")

print(f"  {'#':>3}  {'Insider':15s}  {'Skilled':12s}  {'Average':12s}  "
      f"{'Tr F1':>7}  {'Te F1':>7}  {'Gap':>7}  {'P(Ins)':>7}  {'T10':>4}")
print(f"  {'---':>3}  {'-'*15}  {'-'*12}  {'-'*12}  "
      f"{'-'*7}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*4}")

rows = []
fold_num = 0

for ins, sk, av in product(INSIDERS, SKILLED, AVERAGE):
    fold_num += 1
    test_src  = {ins, sk, av}
    train_src = set(df["source_wallet_name"].unique()) - test_src

    tr = df[df["source_wallet_name"].isin(train_src)].reset_index(drop=True)
    te = df[df["source_wallet_name"].isin(test_src)].reset_index(drop=True)

    X_tr = tr[FEATURES].values.astype(np.float64)
    X_te = te[FEATURES].values.astype(np.float64)
    y_tr = tr["y"].values.astype(int)
    y_te = te["y"].values.astype(int)
    y_tr_bin = tr["y_bin"].values.astype(int)
    y_te_bin = te["y_bin"].values.astype(int)

    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xte_s = sc.transform(X_te)

    m = OrdinalLogisticRegression(l2=L2)
    m.fit(Xtr_s, y_tr)

    # Train performance
    p_tr   = m.p_insider(Xtr_s)
    pd_tr  = (p_tr >= 0.5).astype(int)
    f1_tr  = f1_score(y_tr_bin, pd_tr, zero_division=0)

    # Test performance
    p_te   = m.p_insider(Xte_s)
    pd_te  = (p_te >= 0.5).astype(int)
    prec   = precision_score(y_te_bin, pd_te, zero_division=0)
    rec    = recall_score(y_te_bin, pd_te, zero_division=0)
    f1_te  = f1_score(y_te_bin, pd_te, zero_division=0)
    gap    = f1_tr - f1_te

    # P(Insider) for held-out insider copies
    p_ins_mean = float(p_te[te["label_category"].values == "Insider"].mean())

    # Top-10 precision
    te_sorted = pd.DataFrame({
        "label": te["label_category"].values, "p": p_te
    }).sort_values("p", ascending=False).reset_index(drop=True)
    top10 = int((te_sorted.head(10)["label"] == "Insider").sum())

    print(f"  {fold_num:>3}  {ins:15s}  {sk:12s}  {av:12s}  "
          f"{f1_tr:>7.4f}  {f1_te:>7.4f}  {gap:>+7.4f}  "
          f"{p_ins_mean:>7.4f}  {top10:>3}/10")

    rows.append({
        "fold":         fold_num,
        "test_insider": ins,
        "test_skilled": sk,
        "test_average": av,
        "train_f1":     round(f1_tr, 4),
        "test_prec":    round(prec, 4),
        "test_rec":     round(rec, 4),
        "test_f1":      round(f1_te, 4),
        "gap":          round(gap, 4),
        "p_ins_mean":   round(p_ins_mean, 4),
        "top10":        top10,
    })

results = pd.DataFrame(rows)


# ── Summary by held-out insider ───────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("RESULTS BY HELD-OUT INSIDER (averaged over 9 skilled×average combos)")
print("="*70)
print(f"\n  {'Insider':15s}  {'Train F1':>9}  {'Test F1':>8}  "
      f"{'Gap':>7}  {'P(Ins)':>7}  {'Top-10':>7}")
print(f"  {'-'*15}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
for ins in INSIDERS:
    sub = results[results["test_insider"] == ins]
    print(f"  {ins:15s}  "
          f"{sub['train_f1'].mean():>9.4f}  "
          f"{sub['test_f1'].mean():>8.4f}  "
          f"{sub['gap'].mean():>+7.4f}  "
          f"{sub['p_ins_mean'].mean():>7.4f}  "
          f"{sub['top10'].mean():>5.1f}/10")


# ── Summary by held-out skilled ───────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("RESULTS BY HELD-OUT SKILLED (averaged over 9 insider×average combos)")
print("="*70)
print(f"\n  {'Skilled':12s}  {'Train F1':>9}  {'Test F1':>8}  "
      f"{'Gap':>7}  {'P(Ins)':>7}")
print(f"  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}")
for sk in SKILLED:
    sub = results[results["test_skilled"] == sk]
    print(f"  {sk:12s}  "
          f"{sub['train_f1'].mean():>9.4f}  "
          f"{sub['test_f1'].mean():>8.4f}  "
          f"{sub['gap'].mean():>+7.4f}  "
          f"{sub['p_ins_mean'].mean():>7.4f}")


# ── Summary by held-out average ───────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("RESULTS BY HELD-OUT AVERAGE (averaged over 9 insider×skilled combos)")
print("="*70)
print(f"\n  {'Average':12s}  {'Train F1':>9}  {'Test F1':>8}  "
      f"{'Gap':>7}  {'P(Ins)':>7}")
print(f"  {'-'*12}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}")
for av in AVERAGE:
    sub = results[results["test_average"] == av]
    print(f"  {av:12s}  "
          f"{sub['train_f1'].mean():>9.4f}  "
          f"{sub['test_f1'].mean():>8.4f}  "
          f"{sub['gap'].mean():>+7.4f}  "
          f"{sub['p_ins_mean'].mean():>7.4f}")


# ── Overall summary ───────────────────────────────────────────────────────────

print(f"\n\n{'='*70}")
print("OVERALL SUMMARY — 27-FOLD CV")
print("="*70)

print(f"""
  Mean Train F1:       {results['train_f1'].mean():.4f}  ±  {results['train_f1'].std():.4f}
  Mean Test  F1:       {results['test_f1'].mean():.4f}  ±  {results['test_f1'].std():.4f}
  Mean Gap:            {results['gap'].mean():+.4f}  ±  {results['gap'].std():.4f}
  Mean Test Precision: {results['test_prec'].mean():.4f}  ±  {results['test_prec'].std():.4f}
  Mean Test Recall:    {results['test_rec'].mean():.4f}  ±  {results['test_rec'].std():.4f}
  Mean P(Insider) for held-out insiders: {results['p_ins_mean'].mean():.4f}
  Mean Top-10 insiders caught:           {results['top10'].mean():.2f} / 10

  Distribution of Test F1:
    Min:    {results['test_f1'].min():.4f}
    Median: {results['test_f1'].median():.4f}
    Max:    {results['test_f1'].max():.4f}
    Folds with Test F1 > 0.5:  {(results['test_f1'] > 0.5).sum()} / 27
    Folds with Test F1 = 0.0:  {(results['test_f1'] == 0.0).sum()} / 27
    Folds with Test F1 > 0.9:  {(results['test_f1'] > 0.9).sum()} / 27

  Key finding:
    Train F1 is consistently high ({results['train_f1'].mean():.3f}) — the model fits
    training data well in almost every fold.
    Test F1 varies enormously (std={results['test_f1'].std():.3f}) — performance
    depends entirely on which insider is held out.
    This confirms the issue is heterogeneous insider profiles,
    not model overfitting to noise.
""")

# Best and worst folds
best  = results.nlargest(3, "test_f1")
worst = results[results["test_f1"] == 0.0]

print(f"  Best 3 folds:")
for _, r in best.iterrows():
    print(f"    Fold {int(r['fold']):>2}: {r['test_insider']:15s} + "
          f"{r['test_skilled']:12s} + {r['test_average']:12s}  "
          f"Test F1={r['test_f1']:.4f}")

print(f"\n  Zero-F1 folds ({len(worst)}):")
for _, r in worst.iterrows():
    print(f"    Fold {int(r['fold']):>2}: {r['test_insider']:15s} + "
          f"{r['test_skilled']:12s} + {r['test_average']:12s}  "
          f"P(Ins)={r['p_ins_mean']:.4f}")

print(f"\n  Comparison:")
print(f"    Leaky 80/20:         F1 = 0.807  ← memorization")
print(f"    27-fold mean test:   F1 = {results['test_f1'].mean():.3f}  ← honest generalization")
print(f"    27-fold mean train:  F1 = {results['train_f1'].mean():.3f}  ← model fitting quality")


# ── Save ──────────────────────────────────────────────────────────────────────

results.to_csv(OUT_DIR / "27fold_results.csv", index=False)
print(f"\n✓ Saved to {OUT_DIR}/27fold_results.csv")