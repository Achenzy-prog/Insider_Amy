"""
algo/classifier.py
==================
Stage 4: Claude Haiku batch classifier for insider-prone markets.

Replicates Appendix C, Listing 1 rubric:
  - yes   → market structure makes insider trading plausible
  - no    → market is too liquid/public for information asymmetry
  - unclear → borderline; include in scoring with reduced weight

Only markets classified as 'yes' or 'unclear' have the insider
detection scores applied in Stage 3.
"""

import os, json, time
from pathlib import Path
from typing import Literal

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from config import CLASSIFIER_MODEL, CLASSIFIER_BATCH_SIZE


SYSTEM_PROMPT = """You are an expert in prediction market structure and insider trading analysis.

You will receive a list of Polymarket market descriptions. For each one, classify it as:
  - yes     → the market structure makes insider trading plausible (small pool of people
               with advance knowledge, e.g. internal company data, classified government
               information, pre-announcement corporate decisions)
  - no      → the market is too broad, liquid, or public for meaningful information
               asymmetry (e.g. major sports events, broad macroeconomic outcomes)
  - unclear → borderline cases where insider advantage is possible but uncertain

Return ONLY a JSON array, one object per market, in the same order as input.
Each object: {"index": <int>, "label": "yes"|"no"|"unclear", "reason": "<one sentence>"}

Do not include any text outside the JSON array."""


def classify_markets_batch(
    market_titles: list[str],
    client,
) -> list[dict]:
    """
    Classify a batch of up to 25 market titles.
    Returns list of {index, label, reason} dicts.
    """
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(market_titles))
    user_msg = f"Classify these {len(market_titles)} markets:\n\n{numbered}"

    response = client.messages.create(
        model      = CLASSIFIER_MODEL,
        max_tokens = 1000,
        system     = SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": user_msg}],
    )

    text = response.content[0].text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def classify_all_markets(
    markets: list[dict],
    api_key: str = None,
    batch_size: int = CLASSIFIER_BATCH_SIZE,
) -> list[dict]:
    """
    Classify all markets, batching into groups of `batch_size`.

    markets: list of dicts with at least a 'title' or 'question' field
    Returns: list of {condition_id, title, label, reason} dicts
    """
    if not _ANTHROPIC_AVAILABLE:
        raise ImportError("pip install anthropic to use the Stage 4 classifier")

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = anthropic.Anthropic(api_key=key)

    results = []
    for i in range(0, len(markets), batch_size):
        batch = markets[i : i + batch_size]
        titles = [m.get("title") or m.get("question") or m.get("slug", f"Market {i+j}")
                  for j, m in enumerate(batch)]

        print(f"  Classifying batch {i // batch_size + 1} ({len(titles)} markets)...")
        try:
            classifications = classify_markets_batch(titles, client)
        except Exception as e:
            print(f"  ⚠ Batch failed: {e} — marking as 'unclear'")
            classifications = [{"index": j, "label": "unclear", "reason": "classifier error"}
                               for j in range(len(titles))]

        for cls in classifications:
            j = cls["index"]
            m = batch[j]
            results.append({
                "condition_id": m.get("condition_id") or m.get("conditionId", ""),
                "title":        titles[j],
                "label":        cls.get("label", "unclear"),
                "reason":       cls.get("reason", ""),
            })

        time.sleep(0.5)  # gentle rate limiting

    return results


def filter_to_insider_prone(classified: list[dict]) -> list[dict]:
    """Keep only 'yes' and 'unclear' markets for scoring."""
    return [r for r in classified if r["label"] in ("yes", "unclear")]
