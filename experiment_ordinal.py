"""
Ordinal Logistic Regression on the 900-wallet augmented dataset.

Prints:
  1. Learned coefficients and odds multipliers
  2. LOIO CV results (honest generalization test)
  3. 80/20 stratified split results (precision, recall, F1)

Dataset: data/external/9_wallet_stage_features.parquet
  900 rows — 300 Average, 300 Skilled, 300 Insider
  Round-robin augmented from 9 source wallets (100 copies each)

Ordered classes: Average (0) < Skilled (1) < Insider (2)

Run: python experiment_ordinal.py
"""

import sys, pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

from model import OrdinalLogisticRegression

DATA_PATH = Path("data/external/9_wallet_stage_features.parquet")
OUT_DIR   = Path("data/experiment_ordinal")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    print(f"⚠  {DATA_PATH} not found.")
    sys.exit(1)

FN_RAW   = ["buy_sell_ratio","mean_trade_size_usd","mean_size_vs_daily_vol",
             "mean_dt_hours","num_markets","mean_price_paid"]
FN_CLEAN = ["buy_fraction","mean_trade_size_usd","mean_size_vs_daily",
             "mean_dt_hours","n_markets","mean_price_paid"]
LABEL_MAP       = {"Average": 0, "Skilled": 1, "Insider": 2}
INSIDER_SOURCES = ["AlphaRaccoon", "OmerZiv", "Venezuela"]
L2 = 1.0


# Load

print("Loading 900-wallet augmented dataset...")
raw = pd.read_parquet(DATA_PATH)
raw["y"] = raw["label_category"].map(LABEL_MAP)
df = raw[FN_RAW + ["y","wallet_address","source_wallet_name",
                   "label_category"]].dropna().reset_index(drop=True)
df.columns = FN_CLEAN + ["y","wallet","source_wallet","label"]
for f in FN_CLEAN:
    df[f] = df[f].astype(float)

active = [f for f in FN_CLEAN if df[f].nunique() > 1]
if len(active) < len(FN_CLEAN):
    print(f"  Dropping degenerate: {set(FN_CLEAN)-set(active)}")
df = df[active + ["y","wallet","source_wallet","label"]].reset_index(drop=True)

X_all = df[active].values.astype(np.float64)
y_all = df["y"].values.astype(np.int32)
y_bin = (y_all == 2).astype(int)
src   = df["source_wallet"].values
lbl   = df["label"].values

print(f"  {len(df)} rows  |  {y_bin.sum()} Insider  |  features: {active}\n")


# Section 1: Full model coefficients

print("=" * 70)
print("SECTION 1 — LEARNED COEFFICIENTS (full model, all 900 wallets)")
print("=" * 70)

sc_full    = StandardScaler()
Xa         = sc_full.fit_transform(X_all)
full_model = OrdinalLogisticRegression(l2=L2)
full_model.fit(Xa, y_all)
theta      = full_model.thresholds_

print(f"\n  Decision boundaries:")
print(f"    θ₁ = {theta[0]:.4f}  (Average | Skilled)")
print(f"    θ₂ = {theta[1]:.4f}  (Skilled  | Insider)")
print(f"\n  {'Feature':25s}  {'Coefficient':>12}  {'Odds mult':>10}  Direction")
print(f"  {'-'*25}  {'-'*12}  {'-'*10}  {'-'*22}")

coefs_sorted = sorted(zip(active, full_model.coef_),
                      key=lambda x: abs(x[1]), reverse=True)
for feat, coef in coefs_sorted:
    direction = "↑ more insider-like" if coef > 0 else "↓ more insider-like"
    print(f"  {feat:25s}  {coef:>+12.4f}  {np.exp(coef):>10.2f}x  {direction}")

print(f"\n  Strongest signal: {coefs_sorted[0][0]}  "
      f"(coef={coefs_sorted[0][1]:+.4f}, "
      f"odds ×{np.exp(coefs_sorted[0][1]):.2f} per SD)")
print(f"  Second signal:    {coefs_sorted[1][0]}  "
      f"(coef={coefs_sorted[1][1]:+.4f}, "
      f"odds ×{np.exp(coefs_sorted[1][1]):.2f} per SD)")

pd.DataFrame(coefs_sorted, columns=["feature","coefficient"]).assign(
    odds_multiplier=lambda d: np.exp(d["coefficient"])
).to_csv(OUT_DIR / "coefficients.csv", index=False)


# Section 2: LOIO CV

print(f"\n\n{'='*70}")
print("SECTION 2 — LEAVE-ONE-INSIDER-OUT CROSS-VALIDATION")
print("="*70)
print("\nEach fold holds out all 100 copies of one real insider. Tests whether")
print("the model detects an insider it has NEVER seen during training.\n")

is_ni      = lbl != "Insider"
X_ni       = X_all[is_ni]; y_ni = y_all[is_ni]
fold_rows  = []
all_preds  = []

for held_out in INSIDER_SOURCES:
    is_te  = src == held_out
    is_tri = (lbl == "Insider") & (src != held_out)
    Xte = X_all[is_te]
    Xtr = np.concatenate([X_all[is_tri], X_ni], axis=0)
    ytr = np.concatenate([y_all[is_tri], y_ni])

    sc    = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)
    m     = OrdinalLogisticRegression(l2=L2)
    m.fit(Xtr_s, ytr)
    p_ins = m.p_insider(Xte_s)
    pred  = (p_ins >= 0.5)

    rng   = np.random.RandomState(42)
    ni_i  = rng.choice(len(X_ni), min(100,len(X_ni)), replace=False)
    p_ni  = m.p_insider(sc.transform(X_ni[ni_i]))
    all_p = np.concatenate([p_ins, p_ni])
    ranks = [int(np.sum(all_p >= pi)) for pi in p_ins]
    n10   = sum(r <= 10 for r in ranks)
    n20   = sum(r <= 20 for r in ranks)

    print(f"── {held_out} (n={is_te.sum()}) ──")
    print(f"   Mean P(Insider)   = {p_ins.mean():.4f}  "
          f"{'✓ DETECTED' if p_ins.mean()>0.5 else '✗ MISSED'}")
    print(f"   Median P(Insider) = {np.median(p_ins):.4f}")
    print(f"   Flagged (>0.5)    = {int(pred.sum())}/{len(pred)} ({pred.mean():.1%})")
    print(f"   Top-10 / Top-20   = {n10}/{is_te.sum()}  /  {n20}/{is_te.sum()}\n")

    fold_rows.append({"held_out":held_out,"n_test":int(is_te.sum()),
                      "mean_p":round(float(p_ins.mean()),4),
                      "pct_flagged":round(float(pred.mean()),4),
                      "top10":n10,"top20":n20})
    fold_df = df[is_te][["wallet","source_wallet","label"]].copy()
    fold_df["p_insider"] = p_ins
    fold_df["flagged"]   = pred
    all_preds.append(fold_df)

print("LOIO SUMMARY")
print(f"  {'Insider':15s}  {'Mean P':>7}  {'Flagged':>8}  {'Top-10':>7}  Result")
print(f"  {'-'*15}  {'-'*7}  {'-'*8}  {'-'*7}  {'-'*10}")
for r in fold_rows:
    res = "✓ DETECTED" if r["mean_p"]>0.5 else "✗ MISSED"
    print(f"  {r['held_out']:15s}  {r['mean_p']:>7.4f}  "
          f"{r['pct_flagged']:>8.1%}  {r['top10']:>3}/{r['n_test']:<3}  {res}")

pd.concat(all_preds).to_csv(OUT_DIR/"loio_predictions.csv", index=False)
pd.DataFrame(fold_rows).to_csv(OUT_DIR/"loio_summary.csv", index=False)


# Section 3: 80/20 split

print(f"\n\n{'='*70}")
print("SECTION 3 — 80/20 STRATIFIED SPLIT")
print("="*70)
print("\nNote: augmented copies of the same source wallet may appear in both")
print("train and test. Use LOIO CV above for honest generalization results.")
print("This split shows in-distribution performance.\n")

train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
sc_sp  = StandardScaler()
Xtr_sp = sc_sp.fit_transform(train_df[active].values.astype(np.float64))
Xte_sp = sc_sp.transform(test_df[active].values.astype(np.float64))
ytr_sp = train_df["y"].values.astype(np.int32)
yte_sp = test_df["y"].values.astype(np.int32)

sp_model = OrdinalLogisticRegression(l2=L2)
sp_model.fit(Xtr_sp, ytr_sp)
p_test   = sp_model.p_insider(Xte_sp)
pred_bin = (p_test >= 0.5).astype(int)
yte_bin  = (yte_sp == 2).astype(int)

prec = precision_score(yte_bin, pred_bin, zero_division=0)
rec  = recall_score(yte_bin, pred_bin, zero_division=0)
f1   = f1_score(yte_bin, pred_bin, zero_division=0)

print(f"  Train: {len(train_df)}  |  Test: {len(test_df)}")
print(f"  Precision:  {prec:.4f}")
print(f"  Recall:     {rec:.4f}")
print(f"  F1:         {f1:.4f}")

test_df = test_df.copy()
test_df["p_insider"] = p_test
print(f"\n  Mean P(Insider) by cohort:")
print(test_df.groupby("label")["p_insider"].agg(["mean","median","std"]).round(4).to_string())

test_sorted = test_df.sort_values("p_insider", ascending=False).reset_index(drop=True)
test_sorted["rank"] = test_sorted.index + 1
n_ins_test = int((test_df["label"]=="Insider").sum())
print(f"\n  Precision at top N ({n_ins_test} total insiders in test):")
print(f"  {'Top N':>6}  {'Caught':>6}  {'Precision':>10}")
for k in [5, 10, 20, n_ins_test]:
    caught = int((test_sorted.head(k)["label"]=="Insider").sum())
    print(f"  {k:>6}  {caught:>6}  {caught/k:.0%}")

test_sorted.to_csv(OUT_DIR/"split_test_ranked.csv", index=False)


# Save model for demo

model_dir = Path("data/demo_model")
model_dir.mkdir(parents=True, exist_ok=True)
with open(model_dir/"ordinal_model.pkl","wb") as f: pickle.dump(full_model,f)
with open(model_dir/"scaler.pkl",       "wb") as f: pickle.dump(sc_full,f)
with open(model_dir/"features.pkl",     "wb") as f: pickle.dump(active,f)
with open(model_dir/"train_stats.pkl",  "wb") as f:
    pickle.dump({"mean":sc_full.mean_,"std":sc_full.scale_},f)

print(f"\n✓ Saved to {OUT_DIR}/ and data/demo_model/")