"""Generate the deterministic DEMO comment set at data/sample/youtube_comments.csv.

These comments are SYNTHETIC, template-generated text written for this project. They are not
real YouTube comments and contain no real user data. Output is byte-identical for a given seed.

Usage: python scripts/seed_demo_data.py [--n 300] [--seed 42]
"""

from __future__ import annotations

import argparse
import random
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

SUBJECTS = [
    "this video",
    "the editing",
    "the music",
    "the explanation",
    "this tutorial",
    "the ending",
    "the intro",
    "the camera work",
    "this episode",
    "the soundtrack",
]
POSITIVE = [
    "{s} is really well made, thank you for sharing.",
    "I loved {s}, it made my day.",
    "Honestly {s} was excellent and very helpful.",
    "Great work on {s}, subscribed!",
    "{s} deserves way more views.",
    "This is the best explanation I have found, thanks a lot.",
    "Watched it twice, {s} is fantastic.",
]
NEUTRAL = [
    "What camera did you use for {s}?",
    "Which song is playing at 2:15?",
    "I watched this on my phone during lunch.",
    "Is there a part two planned for {s}?",
    "Uploaded on a Tuesday, interesting.",
    "Can you share the link to the article mentioned?",
    "{s} is about twelve minutes long.",
]
NEGATIVE = [
    "I found {s} boring and too long.",
    "Sadly {s} was disappointing this time.",
    "The audio in {s} is really bad, could not follow.",
    "Not a fan of {s}, the older uploads were better.",
    "Too many ads, I stopped watching halfway.",
    "{s} felt rushed and confusing.",
]
TOXIC = [
    "You are an idiot if you liked {s}.",
    "What a stupid video, the creator is a clown.",
    "Shut up, nobody cares about your dumb opinion.",
    "This is garbage and so are the people defending it.",
]
HINGLISH = [
    "bhai {s} bahut mast hai yaar",
    "kya baat hai, ekdum sahi explanation",
    "yaar ye kya bakwas hai, time waste",
    "bhai music toh kamaal ka hai",
]
HINDI = ["यह वीडियो बहुत अच्छा है", "बहुत बढ़िया जानकारी, धन्यवाद", "मुझे यह पसंद नहीं आया"]
SPANISH = ["Me encantó este video, muy bien explicado", "No me gustó nada, muy aburrido"]

MIX = [
    (POSITIVE, 0.36),
    (NEUTRAL, 0.2),
    (NEGATIVE, 0.2),
    (TOXIC, 0.06),
    (HINGLISH, 0.1),
    (HINDI, 0.05),
    (SPANISH, 0.03),
]


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    pools, weights = zip(*MIX, strict=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    rows = []
    for i in range(n):
        tpl = rng.choice(rng.choices(pools, weights=weights)[0])
        text = tpl.format(s=rng.choice(SUBJECTS))
        text = text[0].upper() + text[1:] if rng.random() < 0.8 else text
        rows.append(
            {
                "comment_id": f"c{i:04d}",
                "text": text,
                "published_at": (start + timedelta(minutes=17 * i)).isoformat(),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    out = ROOT / "data" / "sample" / "youtube_comments.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    generate(args.n, args.seed).to_csv(out, index=False, lineterminator="\n")
    print(f"Wrote {args.n} synthetic demo comments to {out}")
    return out


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
