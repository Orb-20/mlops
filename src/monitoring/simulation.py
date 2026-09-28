"""Deterministic controlled-drift dataset (spec §35).

Models a real shift in how an audience writes: Gen-Z slang, meme phrases, abbreviations and
Hinglish code-mixing, while staying mostly "English". Every text gets an *intended* label by
construction (the template it came from) so the set can be used for reproducible retraining
and held-out evaluation. These are synthetic texts written for this project, not real comments.

Template-level split: each template belongs to exactly one of train/test, so held-out drift
metrics measure generalisation to unseen phrasings, not memorised templates.
"""

from __future__ import annotations

import random

import pandas as pd

# (template, sentiment, toxicity). {s} = subject filler, {e} = emphasis filler.
SUBJECTS = ["this", "the edit", "this song", "the beat", "this video", "the ending", "bro",
            "the vocals", "this collab", "the drop", "this episode", "the thumbnail"]
EMPH = ["fr", "ngl", "lowkey", "highkey", "no cap", "fr fr", "deadass", "istg", "", ""]

TEMPLATES: list[tuple[str, str, str]] = [
    # positive slang
    ("{s} is peak {e}", "positive", "safe"),
    ("absolute cinema {e}", "positive", "safe"),
    ("{s} goes hard {e}", "positive", "safe"),
    ("{s} ate and left no crumbs", "positive", "safe"),
    ("{e} {s} is bussin", "positive", "safe"),
    ("{s} slaps {e}", "positive", "safe"),
    ("W {s}, W creator", "positive", "safe"),
    ("{s} is a whole vibe {e}", "positive", "safe"),
    ("this cooked so hard {e}", "positive", "safe"),
    ("{s} understood the assignment", "positive", "safe"),
    ("goated {s} {e}", "positive", "safe"),
    ("{s} is fire no cap", "positive", "safe"),
    ("bhai {s} toh ekdum mast hai {e}", "positive", "safe"),
    ("kya baat hai yaar, {s} is too good", "positive", "safe"),
    ("{s} ne toh dil jeet liya", "positive", "safe"),
    ("full paisa vasool {s}", "positive", "safe"),
    # negative (non-toxic) slang
    ("{s} is mid {e}", "negative", "safe"),
    ("nah this ain't it {e}", "negative", "safe"),
    ("he fell off {e}", "negative", "safe"),
    ("L {s} tbh", "negative", "safe"),
    ("{s} is giving cringe {e}", "negative", "safe"),
    ("{s} is so washed {e}", "negative", "safe"),
    ("ratio + {s} flopped", "negative", "safe"),
    ("{s} is lowkey boring {e}", "negative", "safe"),
    ("this ain't the one chief", "negative", "safe"),
    ("{s} kinda sus ngl", "negative", "safe"),
    ("yaar {s} bilkul bekar hai", "negative", "safe"),
    ("time waste tha {s}, bakwas", "negative", "safe"),
    ("{s} dekh ke neend aa gayi", "negative", "safe"),
    # neutral slang / questions
    ("who else here in 2026 {e}", "neutral", "safe"),
    ("pov: you came from the reel", "neutral", "safe"),
    ("anyone know the song at 1:20 {e}", "neutral", "safe"),
    ("first comment {e}", "neutral", "safe"),
    ("{s} dropped at 3am lol", "neutral", "safe"),
    ("im just here for the comments", "neutral", "safe"),
    ("bhai ye kaunsa gaana hai", "neutral", "safe"),
    ("koi 2026 mein dekh raha hai?", "neutral", "safe"),
    # toxic slang (directed at people)
    ("ur a clown fr {e}", "negative", "toxic"),
    ("touch grass loser", "negative", "toxic"),
    ("imagine being this dumb lmao", "negative", "toxic"),
    ("shut up bozo nobody asked", "negative", "toxic"),
    ("u r literally braindead {e}", "negative", "toxic"),
    ("stay mad idiot {e}", "negative", "toxic"),
    ("bhai tu pagal hai kya, idiot", "negative", "toxic"),
    ("chup kar bewakoof", "negative", "toxic"),
    ("go cry about it u absolute clown", "negative", "toxic"),
    ("ratio + ur mom + stupid", "negative", "toxic"),
]


def _render(tpl: str, rng: random.Random) -> str:
    t = tpl.format(s=rng.choice(SUBJECTS), e=rng.choice(EMPH))
    t = " ".join(t.split())
    if rng.random() < 0.3:
        t += rng.choice([" 💀", " 🔥", " 😭", " 😂", "!!", " lol"])
    return t


def template_split(seed: int, test_frac: float = 0.3) -> dict[int, str]:
    """Stratified by (sentiment, toxicity): each group contributes templates to both splits."""
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[int]] = {}
    for i, (_, s, t) in enumerate(TEMPLATES):
        groups.setdefault((s, t), []).append(i)
    assign: dict[int, str] = {}
    for ids in groups.values():
        ids = ids[:]
        rng.shuffle(ids)
        k = max(1, round(len(ids) * test_frac))
        for j, i in enumerate(ids):
            assign[i] = "test" if j < k else "train"
    return assign


def build_drift_dataset(n_train: int = 1200, n_test: int = 400, n_window: int = 300,
                        seed: int = 42) -> dict[str, pd.DataFrame]:
    """Returns train/test (labelled) and window (unlabelled texts that simulate a new stream).
    The monitoring window uses templates from both splits, like a real incoming stream."""
    rng = random.Random(seed)
    assign = template_split(seed)
    pools = {sp: [i for i, v in assign.items() if v == sp] for sp in ("train", "test")}

    def make(split: str, n: int, ids: list[int]) -> pd.DataFrame:
        rows, seen = [], set()
        attempts = 0
        while len(rows) < n and attempts < n * 50:
            attempts += 1
            i = rng.choice(ids)
            text = _render(TEMPLATES[i][0], rng)
            if text in seen:
                continue
            seen.add(text)
            rows.append({"text": text, "sentiment": TEMPLATES[i][1],
                         "toxicity": TEMPLATES[i][2], "template_id": i, "split": split})
        return pd.DataFrame(rows)

    train = make("train", n_train, pools["train"])
    test = make("test", n_test, pools["test"])
    test = test[~test["text"].isin(set(train["text"]))].reset_index(drop=True)
    window = make("window", n_window, list(range(len(TEMPLATES))))
    return {"train": train, "test": test, "window": window}


SENT_ID = {"negative": 0, "neutral": 1, "positive": 2}
TOX_ID = {"safe": 0, "toxic": 1}


def as_task_frames(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert to the {task: DataFrame[text,label]} format used by training/evaluation."""
    return {
        "sentiment": pd.DataFrame({"text": df["text"], "label": df["sentiment"].map(SENT_ID)}),
        "toxicity": pd.DataFrame({"text": df["text"], "label": df["toxicity"].map(TOX_ID)}),
    }
