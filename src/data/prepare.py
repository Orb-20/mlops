"""Phase 2 data pipeline: download → normalise → validate → split → write processed parquet.

Usage:
    python -m src.data.prepare            # download (cached in data/raw) and build splits
    python -m src.data.prepare --offline  # rebuild from data/raw only
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.settings import get_config, project_path, set_seed
from src.data.preprocessing import clean_frame
from src.data.schemas import SENTIMENT_LABELS, TOXICITY_LABELS
from src.data.validation import check_no_split_leakage, validate_labeled_frame
from src.data.versioning import write_manifest

log = logging.getLogger(__name__)


def _hf_download(repo_id: str, filename: str, dest: Path) -> Path:
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        log.info("Using cached %s", dest)
        return dest
    log.info("Downloading %s/%s", repo_id, filename)
    src = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    shutil.copy(src, dest)
    return dest


def download_raw(offline: bool = False) -> dict[str, Path]:
    cfg = get_config()["data"]
    raw = project_path("data", "raw")
    s = cfg["sentiment"]
    files = {
        f"sentiment_{split}": raw / "sentiment" / f"{split}.parquet"
        for split in ("train", "validation", "test")
    }
    files["toxicity"] = raw / "toxicity" / "train.csv"
    if offline:
        missing = [str(p) for p in files.values() if not p.exists()]
        if missing:
            raise FileNotFoundError(f"--offline but raw files missing: {missing}")
        return files
    for split in ("train", "validation", "test"):
        _hf_download(
            s["hf_dataset"],
            f"{s['hf_config']}/{split}-00000-of-00001.parquet",
            files[f"sentiment_{split}"],
        )
    t = cfg["toxicity"]
    _hf_download(t["hf_dataset"], t["hf_file"], files["toxicity"])
    return files


def _sample(df: pd.DataFrame, n: int | None, seed: int, stratify: str = "label") -> pd.DataFrame:
    if n is None or len(df) <= n:
        return df
    sampled, _ = train_test_split(df, train_size=n, random_state=seed, stratify=df[stratify])
    return sampled.reset_index(drop=True)


def build_sentiment(files: dict[str, Path], seed: int) -> dict[str, pd.DataFrame]:
    cfg = get_config()["data"]
    s, tcfg = cfg["sentiment"], cfg["text"]
    caps = {"train": s["max_train"], "val": s["max_val"], "test": s["max_test"]}
    out = {}
    seen: set[str] = set()
    # test first, so leakage removal never shrinks the evaluation set
    for split, key in (
        ("test", "sentiment_test"),
        ("val", "sentiment_validation"),
        ("train", "sentiment_train"),
    ):
        df = pd.read_parquet(files[key])[["text", "label"]]
        df = clean_frame(df, min_chars=tcfg["min_chars"], max_chars=tcfg["max_chars"])
        df = df[~df["text"].isin(seen)]
        df = _sample(df, caps[split], seed)
        seen |= set(df["text"])
        out[split] = df.assign(label=df["label"].astype(int)).reset_index(drop=True)
    return out


def build_toxicity(files: dict[str, Path], seed: int) -> dict[str, pd.DataFrame]:
    cfg = get_config()["data"]
    t, tcfg = cfg["toxicity"], cfg["text"]
    raw_cols = t["raw_label_columns"]
    df = pd.read_csv(files["toxicity"])
    df = df.rename(columns={"comment_text": "text"})
    df["label"] = (df[raw_cols].max(axis=1) > 0).astype(int)
    df = df[["text", "label", *raw_cols]]
    df = clean_frame(df, min_chars=tcfg["min_chars"], max_chars=tcfg["max_chars"])
    df = _sample(df, t["max_total"], seed)
    sp = t["split"]
    train, rest = train_test_split(
        df, train_size=sp["train"], random_state=seed, stratify=df["label"]
    )
    val_frac = sp["val"] / (sp["val"] + sp["test"])
    val, test = train_test_split(
        rest, train_size=val_frac, random_state=seed, stratify=rest["label"]
    )
    return {
        k: v.reset_index(drop=True) for k, v in {"train": train, "val": val, "test": test}.items()
    }


def prepare(offline: bool = False) -> Path:
    seed = set_seed()
    files = download_raw(offline=offline)
    out_dir = project_path("data", "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict] = {}
    for task, builder, n_classes, names in (
        ("sentiment", build_sentiment, len(SENTIMENT_LABELS), SENTIMENT_LABELS),
        ("toxicity", build_toxicity, len(TOXICITY_LABELS), TOXICITY_LABELS),
    ):
        splits = builder(files, seed)
        leakage = check_no_split_leakage(splits)
        if leakage:
            raise ValueError(f"{task}: split leakage: {leakage}")
        for split, df in splits.items():
            validate_labeled_frame(df, f"{task}/{split}", n_classes, min_rows=50).raise_if_failed()
            df.to_parquet(out_dir / f"{task}_{split}.parquet", index=False)
            dist = df["label"].value_counts(normalize=True).sort_index()
            stats[f"{task}_{split}"] = {
                "rows": len(df),
                "label_dist": {names[i]: round(float(p), 4) for i, p in dist.items()},
            }
            log.info("%s/%s: %d rows %s", task, split, len(df), stats[f"{task}_{split}"])

    manifest = write_manifest(out_dir, {"seed": seed, "splits": stats})
    log.info("Wrote %s", manifest)
    return out_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument(
        "--step",
        choices=["download", "build", "all"],
        default="all",
        help="download: fetch raw only; build: raw -> processed (offline)",
    )
    args = ap.parse_args()
    if args.step == "download":
        download_raw()
    else:
        prepare(offline=args.offline or args.step == "build")


if __name__ == "__main__":
    main()
