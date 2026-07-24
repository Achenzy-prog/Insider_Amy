"""
agent_labeler.py
================
Uses an LLM agent via OpenAI-compatible endpoint to generate pseudo-labels
for wallets, producing a weakly-supervised dataset for comparing against
our quantitative model outputs.

Requires .env file with:
  OPENAI_API_KEY=your-key-here

Run: python agent_labeler.py
"""

import os, json, time, sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

import pandas as pd
import numpy as np

load_dotenv()

OUT_DIR = Path("data/agent_labels")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("data/external/wallet_stage_features.parquet")
if not DATA_PATH.exists():
    print(f"⚠ {DATA_PATH} not found.")
    print("  Copy wallet_stage_features.parquet to data/external/")
    sys.exit(1)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://code.newcli.com/codex/v1"
)

BATCH_SIZE = 20


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert analyst specializing in detecting insider trading on Polymarket, a decentralized prediction markets platform where users trade binary outcome tokens (YES/NO) on future events.

## YOUR TASK
You will receive batches of wallet feature vectors. For each wallet, you must assign a suspicion label and explain your reasoning based on the feature values and the domain knowledge below.

---

## DOMAIN KNOWLEDGE: WHAT INSIDER TRADING LOOKS LIKE ON POLYMARKET

### Core thesis
Insider traders appear different because of their EVENT-LEVEL BEHAVIOR, not because they make money overall. A genuinely skilled trader performs consistently across many market types. An insider only has an edge in specific markets where they have non-public information — so their portfolio looks narrow, concentrated, and timed around key information events.

### Known confirmed cases (ground truth)
Three wallets are confirmed insiders from legal cases and press reporting:

1. **AlphaRaccoon (Google Year in Search 2025)**
   - Michele Spagnuolo, a Google employee, traded on Google's internal "Year in Search" ranking data before public release
   - Made 22/23 correct predictions, $1.2M profit
   - Key behavior: very high buy fraction (accumulated YES positions), moderate trade sizes, early entry before price jumps, but MODERATE market count (92 markets total across full history — diluted by legitimate trading outside the insider window)
   - Feature signature: high buy_fraction, moderate mean_dt_hours (negative = early), moderate n_markets

2. **OmerZiv (Iran-Israel conflict)**
   - Traded on Iran/Israel military conflict outcomes with suspected advance knowledge
   - 5/5 perfect wins
   - Key behavior: extremely concentrated (only 5 markets ever), very high win rate, large position sizes relative to market liquidity
   - Feature signature: very low n_markets, high buy_fraction, high mean_size_vs_daily

3. **Venezuela/Maduro**
   - US soldier with advance knowledge of Maduro capture operation
   - 4/4 wins, extremely concentrated
   - Key behavior: burner account (new account, few markets), large positions for a new wallet
   - Feature signature: very low n_markets (4), high buy_fraction

### Behavioral signals and their importance (from our logistic regression model)
These are the learned log-odds coefficients — the larger the absolute value, the stronger the signal:

| Feature | Coefficient | Interpretation |
|---------|-------------|----------------|
| n_markets | -2.93 | STRONGEST SIGNAL. Fewer markets = more suspicious. Insiders concentrate; skilled traders diversify. A wallet trading <20 markets with high returns is very suspicious. |
| buy_fraction | +2.41 | STRONG SIGNAL. High buy rate = accumulating positions with conviction before resolution. Insiders know which side wins, so they buy heavily and don't hedge. |
| mean_trade_size_usd | +1.23 | Larger trades = more suspicious. Insiders bet big when they have an edge. BUT: also triggered by whales (large legitimate traders). Must be evaluated with other signals. |
| mean_size_vs_daily | -1.25 | Counter-intuitive: lower ratio = more suspicious in our data. May reflect insiders trading on larger-volume cover markets. Interpret carefully. |
| mean_dt_hours | -0.76 | More negative = traded further before the next price jump = more suspicious. Insiders enter before markets move. Values below -10 hours are notable. |
| mean_price_paid | +0.30 | Weakest signal. Slightly higher entry price = more suspicious. Noise-level signal; do not weight heavily. |

### Key distinctions: insider vs skilled vs whale
- **Insider**: few markets, high buy fraction, early timing, wins concentrated in one event category, often a new or rarely-used account around the event
- **Skilled**: many markets (broad diversification), consistent performance across categories, win rate elevated but not extreme, timing not systematically early
- **Whale**: large trade sizes and volumes, but DIVERSIFIED across many markets, average or below-average timing, average win rate relative to market

### Red flags to look for
- n_markets < 20 combined with high buy_fraction (>0.7) → very suspicious
- mean_dt_hours very negative (< -20hr) → systematically entering before market-moving events
- mean_trade_size_usd extremely high (>$1000 average per trade) with low n_markets → concentrated large bet
- buy_fraction = 1.0 or near 1.0 → only buying, never selling, very high conviction

### False positive patterns (wallets that LOOK like insiders but aren't)
- High mean_trade_size_usd alone → could be a whale, not an insider. Check n_markets: if >100, probably just a large legitimate trader
- buy_fraction near 1.0 with tiny trade sizes → may just be a casual trader who bought and held without selling, not an insider
- low n_markets with low buy_fraction → could be an inactive/new wallet, not necessarily suspicious

---

## OUTPUT FORMAT
You must respond with ONLY a valid JSON array. No preamble, no explanation outside the JSON.
Each element has exactly these fields:
{
  "wallet_index": <integer, the index provided>,
  "label": <one of: "confirmed_insider", "suspected_insider", "informed_trader", "skilled_trader", "whale", "normal_trader">,
  "confidence": <one of: "high", "medium", "low">,
  "suspicion_score": <float between 0.0 and 1.0, where 1.0 = certain insider>,
  "primary_signal": <the single feature that most drove your decision>,
  "reasoning": <2-3 sentences explaining your reasoning, citing specific feature values>
}

Label definitions:
- confirmed_insider: would refer for immediate investigation; strong multi-signal case
- suspected_insider: enough signals to flag; warrants deeper review
- informed_trader: may have soft information advantages (e.g. expert in domain); not clearly illegal
- skilled_trader: legitimately skilled; high performance explained by skill/strategy
- whale: large legitimate trader; high volume but diversified and not suspicious
- normal_trader: no notable signals; baseline behavior
"""


# ── Load and prepare wallet features ──────────────────────────────────────────

print("Loading wallet features...")
raw = pd.read_parquet(DATA_PATH)

FEATURE_MAP = {
    "buy_sell_ratio":         "buy_fraction",
    "mean_trade_size_usd":    "mean_trade_size_usd",
    "mean_size_vs_daily_vol": "mean_size_vs_daily",
    "mean_dt_hours":          "mean_dt_hours",
    "num_markets":            "n_markets",
    "mean_price_paid":        "mean_price_paid",
}

df = raw.rename(columns={
    "wallet_address":     "wallet",
    "label_category":     "true_label",
    "source_wallet_name": "source_wallet",
    **FEATURE_MAP,
})

FEATURES = list(FEATURE_MAP.values())
df_clean = df[["wallet","true_label","source_wallet"] + FEATURES].dropna().reset_index(drop=True)

print(f"Wallets to label: {len(df_clean)}")
print(f"True label distribution: {df_clean['true_label'].value_counts().to_dict()}")


# ── Helper: format a batch for the agent ──────────────────────────────────────

def format_batch(batch_df: pd.DataFrame, start_idx: int) -> str:
    lines = [
        f"Label these {len(batch_df)} wallets. Each row is one wallet's aggregated trading features.",
        "",
        f"{'Index':>5}  {'buy_fraction':>12}  {'trade_size_usd':>14}  {'size_vs_daily':>13}  "
        f"{'dt_hours':>9}  {'n_markets':>9}  {'price_paid':>10}",
        f"{'-----':>5}  {'------------':>12}  {'--------------':>14}  {'-------------':>13}  "
        f"{'---------':>9}  {'---------':>9}  {'----------':>10}",
    ]
    for i, (_, row) in enumerate(batch_df.iterrows()):
        idx = start_idx + i
        lines.append(
            f"{idx:>5}  "
            f"{row['buy_fraction']:>12.4f}  "
            f"{row['mean_trade_size_usd']:>14.2f}  "
            f"{row['mean_size_vs_daily']:>13.6f}  "
            f"{row['mean_dt_hours']:>9.2f}  "
            f"{row['n_markets']:>9.0f}  "
            f"{row['mean_price_paid']:>10.4f}"
        )
    return "\n".join(lines)


# ── Call agent in batches ──────────────────────────────────────────────────────

results  = []
n_total  = len(df_clean)

print(f"\nSending {n_total} wallets to agent in batches of {BATCH_SIZE}...")
print(f"Estimated API calls: {(n_total + BATCH_SIZE - 1) // BATCH_SIZE}\n")

for batch_start in range(0, n_total, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n_total)
    batch_df  = df_clean.iloc[batch_start:batch_end]
    batch_num = batch_start // BATCH_SIZE + 1

    print(f"  Batch {batch_num}: wallets {batch_start}–{batch_end-1}...", end=" ", flush=True)

    user_message = format_batch(batch_df, batch_start)

    try:
        response = client.chat.completions.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
        )
        text = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.split("```")[0].strip()

        batch_labels = json.loads(text)
        results.extend(batch_labels)
        print(f"✓ ({len(batch_labels)} labeled)")

    except json.JSONDecodeError as e:
        print(f"✗ JSON parse error: {e}")
        for i in range(batch_start, batch_end):
            results.append({
                "wallet_index":    i,
                "label":           "unknown",
                "confidence":      "low",
                "suspicion_score": 0.5,
                "primary_signal":  "parse_error",
                "reasoning":       "Batch failed to parse."
            })
    except Exception as e:
        print(f"✗ API error: {e}")
        for i in range(batch_start, batch_end):
            results.append({
                "wallet_index":    i,
                "label":           "unknown",
                "confidence":      "low",
                "suspicion_score": 0.5,
                "primary_signal":  "api_error",
                "reasoning":       str(e)
            })

    time.sleep(0.3)


# ── Merge agent labels with wallet features ────────────────────────────────────

print(f"\nMerging results...")
labels_df = pd.DataFrame(results).sort_values("wallet_index").reset_index(drop=True)

df_clean["wallet_index"] = df_clean.index
output = df_clean.merge(labels_df, on="wallet_index", how="left")

LABEL_SUSPICION = {
    "confirmed_insider": 1.0,
    "suspected_insider": 0.8,
    "informed_trader":   0.6,
    "skilled_trader":    0.3,
    "whale":             0.2,
    "normal_trader":     0.1,
    "unknown":           0.5,
}
output["agent_suspicion_numeric"] = output["label"].map(LABEL_SUSPICION).fillna(0.5)


# ── Summary ───────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("AGENT LABELING SUMMARY")
print("="*70)

print(f"\nAgent label distribution:")
print(output["label"].value_counts().to_string())

print(f"\nAgent labels vs true labels:")
print(pd.crosstab(output["true_label"], output["label"]).to_string())

print(f"\nMean agent suspicion score by true label:")
print(output.groupby("true_label")["agent_suspicion_numeric"].agg(
    ["mean","median","std"]
).round(3).to_string())

insider_mask  = output["true_label"] == "Insider"
agent_caught  = output[insider_mask]["label"].isin(
    ["confirmed_insider","suspected_insider","informed_trader"]
)
print(f"\nOf {insider_mask.sum()} true Insider wallets:")
print(f"  Agent flagged as suspicious: {agent_caught.sum()} ({agent_caught.mean():.1%})")
print(f"  Agent label breakdown:")
print(output[insider_mask]["label"].value_counts().to_string())


# ── Save ──────────────────────────────────────────────────────────────────────

output.to_csv(OUT_DIR / "agent_labeled_wallets.csv", index=False)
print(f"\n✓ Saved to {OUT_DIR}/agent_labeled_wallets.csv")
print(f"  {len(output)} wallets with agent labels + true labels + features")
print(f"\nCompare 'agent_suspicion_numeric' vs model P(Insider) from")
print(f"experiment_classification.py to see where the two methods agree.")