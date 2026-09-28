"""Bootstrap a fresh environment: demo data → processed data → first model → gated promotion.

Idempotent: skips training if a production model already exists (use --force to retrain).

Usage: python scripts/setup_local.py [--max-train 8000] [--epochs 2] [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("setup_local")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-train", type=int, help="rows per task (default: configs/model.yaml)")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    from scripts.seed_demo_data import main as seed_demo
    from src.data.prepare import prepare
    from src.database.connection import init_db
    from src.evaluation.promotion import run_promotion
    from src.training import registry
    from src.training.train import train_model

    init_db()
    if not (ROOT / "data" / "sample" / "youtube_comments.csv").exists():
        seed_demo([])
    if not (ROOT / "data" / "processed" / "manifest.json").exists():
        log.info("Preparing datasets (downloads public data from Hugging Face)")
        prepare()
    if registry.get_version_by_alias("production") is not None and not args.force:
        log.info("A production model already exists; nothing to do (use --force to retrain).")
        return 0
    overrides = {}
    if args.max_train:
        overrides["max_train_per_task"] = args.max_train
    if args.epochs:
        overrides["epochs"] = args.epochs
    res = train_model(overrides=overrides, tags={"trigger": "bootstrap"})
    decision = run_promotion(res.version)
    log.info("Bootstrap model v%s: %s %s", res.version, decision.decision, decision.reasons)
    if decision.decision != "PROMOTE":
        log.error(
            "The bootstrap model did not pass the gates (see reasons above). Increase "
            "--max-train/--epochs; the gates in configs/gates.yaml are not bypassed."
        )
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
