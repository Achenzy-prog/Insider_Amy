"""
experiment_grouped_split.py
============================
Grouped train/test split: 3 folds, one per insider.

In each fold:
  Test set:  all copies of 1 insider + 1 skilled + 1 average (300 wallets)
  Train set: remaining 6 source wallets (600 wallets)

This ensures:
  - No augmented copies of the same source wallet in both train and test
  - The test set always contains exactly one of each class
  - We can measure precision/recall/F1 meaningfully per fold

Fold assignment:
  Fold 1 (AlphaRaccoon): test = AlphaRaccoon + Gambler2026 + OTW10T
  Fold 2 (OmerZiv):      test = OmerZiv      + huludubu    + taerv534
  Fold 3 (Venezuela):    test = Venezuela    + Betwick     + Minecraft

The skilled and average wallets were assigned round-robin to match
the insider fold so each non-insider wallet appears in exactly one
test fold.

Dataset: data/external/9_wallet_stage_features.parquet

Run: python experiment_grouped_split.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from model import OrdinalLogisticRegression

DATA_PATH = Path("data/external/9_wallet_stage_features.parquet")
OUT_DIR   = Path("data/experiment_grouped_split")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    print(f"⚠  {DATA_PATH} not found.")
    sys.exit(1)

FEATURES = ['buy_fraction','mean_trade_size_usd','mean_size_vs_daily',
            'mean_dt_hours','n_markets','mean_price_paid']
LABEL_MAP = {"Average": 0, "Skilled": 1, "Insider": 2}
L2 = 1.0

# Fold definitions: each fold leaves out 1 insider + 1 skilled + 1 average
# Round-robin assignment so every source wallet appears in exactly one test fold
FOLDS = [
    {
        "name":    "Fold 1 — AlphaRaccoon",
        "insider": "AlphaRaccoon",
        "skilled": "Gambler2026",
        "average": "OTW10T",
    },
    {
        "name":    "Fold 2 — OmerZiv",
        "insider": "OmerZiv",
        "skilled": "huludubu",
        "average": "taerv534",
    },
    {
        "name":    "Fold 3 — Venezuela",
        "insider": "Venezuela",
        "skilled": "Betwick",
        "average": "Minecraft",
    },
]


# ── Load ──────────────────────────────────────────────────────────────────────

print("Loading 900-wallet augmented dataset...")
raw = pd.read_parquet(DATA_PATH)
raw = raw.rename(columns={
    "buy_sell_ratio":         "buy_fraction",
    "mean_size_vs_daily_vol": "mean_size_vs_daily",
    "num_markets":            "n_markets",
})
raw["y"]   = raw["label_category"].map(LABEL_MAP)
raw["y_bin"] = (raw["label_category"] == "Insider").astype(int)

for f in FEATURES:
    raw[f] = raw[f].astype(float)

df = raw[FEATURES + ["y","y_bin","source_wallet_name","label_category"]].dropna().reset_index(drop=True)

print(f"  {len(df)} rows after dropna")
print(f"  Source wallets: {sorted(df['source_wallet_name'].unique())}\n")


# ── Run 3 folds ───────────────────────────────────────────────────────────────

print("=" * 70)
print("GROUPED TRAIN/TEST SPLIT — 3 FOLDS")
print("=" * 70)
print("""
Each fold leaves out 1 insider + 1 skilled + 1 average wallet (all copies).
Train: 6 source wallets × 100 copies = 600 wallets
Test:  3 source wallets × 100 copies = 300 wallets (100 per class)

This is leak-free: no augmented copies of the same source wallet
appear in both train and test.
""")

fold_rows  = []
all_preds  = []

for fold in FOLDS:
    test_sources  = {fold["insider"], fold["skilled"], fold["average"]}
    train_sources = set(df["source_wallet_name"].unique()) - test_sources

    train_df = df[df["source_wallet_name"].isin(train_sources)].reset_index(drop=True)
    test_df  = df[df["source_wallet_name"].isin(test_sources)].reset_index(drop=True)

    X_tr = train_df[FEATURES].values.astype(np.float64)
    X_te = test_df[FEATURES].values.astype(np.float64)
    y_tr = train_df["y"].values.astype(int)
    y_te = test_df["y"].values.astype(int)
    y_te_bin = test_df["y_bin"].values.astype(int)

    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(X_tr)
    Xte_s = sc.transform(X_te)

    model = OrdinalLogisticRegression(l2=L2)
    model.fit(Xtr_s, y_tr)

    p_ins = model.p_insider(Xte_s)
    pred  = (p_ins >= 0.5).astype(int)

    # ── Train set performance ────────────────────────────────────────────
    p_tr      = model.p_insider(Xtr_s)
    pred_tr   = (p_tr >= 0.5).astype(int)
    y_tr_bin  = train_df["y_bin"].values.astype(int)
    prec_tr   = precision_score(y_tr_bin, pred_tr, zero_division=0)
    rec_tr    = recall_score(y_tr_bin, pred_tr, zero_division=0)
    f1_tr     = f1_score(y_tr_bin, pred_tr, zero_division=0)

    # ── Test set performance ──────────────────────────────────────────────
    prec = precision_score(y_te_bin, pred, zero_division=0)
    rec  = recall_score(y_te_bin, pred, zero_division=0)
    f1   = f1_score(y_te_bin, pred, zero_division=0)

    # Per-wallet-type P(Insider) stats
    test_df = test_df.copy()
    test_df["p_insider"] = p_ins
    test_df["flagged"]   = pred

    p_by_type = test_df.groupby("label_category")["p_insider"].agg(
        ["mean","median"]
    ).round(4)

    # Ranking within test set
    test_sorted = test_df.sort_values("p_insider", ascending=False).reset_index(drop=True)
    test_sorted["rank"] = test_sorted.index + 1
    n_ins_test = int(y_te_bin.sum())

    print(f"── {fold['name']} ──")
    print(f"   Train: {sorted(train_sources)}")
    print(f"   Test:  {fold['insider']} (Insider) + "
          f"{fold['skilled']} (Skilled) + {fold['average']} (Average)")
    print(f"   Train size: {len(train_df)}  |  Test size: {len(test_df)}")
    print()
    print(f"   Train set performance:")
    print(f"     Precision: {prec_tr:.4f}")
    print(f"     Recall:    {rec_tr:.4f}")
    print(f"     F1:        {f1_tr:.4f}")
    print(f"     P(Insider) flagged in train: "
          f"{pred_tr[y_tr_bin==1].mean():.1%} of true insiders, "
          f"{pred_tr[y_tr_bin==0].mean():.1%} of non-insiders")
    print()
    print(f"   Test set performance:")
    print(f"     Precision: {prec:.4f}")
    print(f"     Recall:    {rec:.4f}")
    print(f"     F1:        {f1:.4f}")
    print()
    print(f"   Mean P(Insider) by cohort:")
    print(p_by_type.to_string())
    print()

    # Precision at top N
    print(f"   Precision at top N ({n_ins_test} true insiders in test):")
    print(f"   {'Top N':>6}  {'Caught':>6}  {'Precision':>10}")
    for k in [5, 10, 20, n_ins_test]:
        caught = int((test_sorted.head(k)["label_category"]=="Insider").sum())
        print(f"   {k:>6}  {caught:>6}  {caught/k:.0%}")
    print()

    # Flagging rate per wallet type
    flag_by_type = test_df.groupby("label_category")["flagged"].mean().round(4)
    print(f"   Flagged (P>0.5) by cohort:")
    print(flag_by_type.to_string())
    print()

    # Full ranking (top 20)
    print(f"   Top 20 wallets by P(Insider):")
    print(f"   {'Rank':>4}  {'Source':>15}  {'Label':>8}  {'P(Insider)':>10}  Correct?")
    print(f"   {'----':>4}  {'-'*15}  {'-------':>8}  {'----------':>10}  --------")
    for _, row in test_sorted.head(20).iterrows():
        marker = "✓ INSIDER" if row["label_category"]=="Insider" else ""
        print(f"   {int(row['rank']):>4}  {row['source_wallet_name']:>15}  "
              f"{row['label_category']:>8}  {row['p_insider']:>10.4f}  {marker}")
    print()

    fold_rows.append({
        "fold":          fold["name"],
        "test_insider":  fold["insider"],
        "test_skilled":  fold["skilled"],
        "test_average":  fold["average"],
        "n_train":       len(train_df),
        "n_test":        len(test_df),
        "train_prec":    round(prec_tr, 4),
        "train_rec":     round(rec_tr, 4),
        "train_f1":      round(f1_tr, 4),
        "test_prec":     round(prec, 4),
        "test_rec":      round(rec, 4),
        "test_f1":       round(f1, 4),
        "p_ins_mean":    round(float(test_df[test_df["label_category"]=="Insider"]["p_insider"].mean()), 4),
        "top10_caught":  int((test_sorted.head(10)["label_category"]=="Insider").sum()),
    })

    all_preds.append(test_df)


# ── Summary ───────────────────────────────────────────────────────────────────

print("=" * 70)
print("SUMMARY ACROSS ALL 3 FOLDS")
print("=" * 70)
summary = pd.DataFrame(fold_rows)
print(f"\n  {'Fold':30s}  {'Train F1':>9}  {'Test F1':>8}  {'Gap':>7}  {'P(Ins)':>7}  {'Top-10':>7}")
print(f"  {'-'*30}  {'-'*9}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*7}")
for _, r in summary.iterrows():
    gap = r["train_f1"] - r["test_f1"]
    print(f"  {r["fold"]:30s}  {r["train_f1"]:>9.4f}  {r["test_f1"]:>8.4f}  "
          f"{gap:>+7.4f}  {r["p_ins_mean"]:>7.4f}  "
          f"{r["top10_caught"]:>3}/100")

print(f"\n  Mean across folds:")
print(f"    Train F1:  {summary["train_f1"].mean():.4f}")
print(f"    Test F1:   {summary["test_f1"].mean():.4f}")
print(f"    Gap:       {(summary["train_f1"]-summary["test_f1"]).mean():+.4f}")
print(f"    P(Insider) for held-out insiders: {summary["p_ins_mean"].mean():.4f}")

print(f"""
Key comparisons:
  Leaky 80/20 (test only):   F1=0.807  ← inflated, memorization
  Grouped split (train mean): F1={summary['train_f1'].mean():.3f}  ← how well model fits training data
  Grouped split (test mean):  F1={summary['test_f1'].mean():.3f}  ← honest generalization
  Generalization gap:         {(summary['train_f1']-summary['test_f1']).mean():+.3f}

A large train/test gap within a fold means the model fit the training
wallets well but failed to generalize to the test wallets — confirming
the poor generalization is not because the model failed to learn, but
because what it learned does not transfer across insider types.
""")


# ── Save ──────────────────────────────────────────────────────────────────────

summary.to_csv(OUT_DIR / "fold_summary.csv", index=False)
pd.concat(all_preds).to_csv(OUT_DIR / "all_predictions.csv", index=False)
print(f"✓ Saved to {OUT_DIR}/")
print("  fold_summary.csv    — precision/recall/F1 per fold")
print("  all_predictions.csv — per-wallet predictions across all folds")