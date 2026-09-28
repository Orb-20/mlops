"""Deterministic drift demonstration (spec §35). Works without any external API.

1. load the production model's reference window
2. build the controlled shifted dataset (slang / meme / Hinglish; labels by construction)
3. STABLE window  = held-out sample of the training distribution  -> expect no drift
4. SHIFTED window = unlabelled stream of shifted comments          -> expect DRIFT DETECTED
5. persist both as drift events, write reports/drift/drift_report_<ts>.{json,md}
6. trigger the retraining pipeline for the drift event (unless --no-retrain)

Usage:
    python scripts/simulate_drift.py                  # detect + retrain
    python scripts/simulate_drift.py --no-retrain     # detect only
    python scripts/simulate_drift.py --build-only     # just (re)build data/drift (DVC stage)
    python scripts/simulate_drift.py --scenario corrupted_labels   # retrain with bad labels
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.settings import get_config, project_path  # noqa: E402
from src.data.schemas import CommentRecord, utcnow  # noqa: E402
from src.data.versioning import write_manifest  # noqa: E402
from src.monitoring.reference import reference_texts  # noqa: E402
from src.monitoring.simulation import build_drift_dataset  # noqa: E402

log = logging.getLogger("simulate_drift")


def build_data() -> dict:
    seed = get_config()["project"]["seed"]
    data = build_drift_dataset(seed=seed)
    out = project_path("data", "drift")
    out.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    write_manifest(out, {"seed": seed, "rows": {k: len(v) for k, v in data.items()},
                         "note": "synthetic controlled drift data; labels by construction"})
    log.info("Drift dataset written to %s: %s", out, {k: len(v) for k, v in data.items()})
    return data


def _records(texts: list[str], content_id: str) -> list[CommentRecord]:
    t0 = utcnow()
    return [CommentRecord(source="simulation", source_content_id=content_id,
                          source_comment_id=f"{content_id}:{i:05d}", text=t,
                          timestamp=t0 + timedelta(seconds=i)) for i, t in enumerate(texts)]


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--no-retrain", action="store_true")
    ap.add_argument("--scenario", default="normal")
    args = ap.parse_args(argv)

    data = build_data()
    if args.build_only:
        return 0

    from src.database.connection import init_db
    from src.pipeline.analyze import process_comments

    init_db()
    stamp = utcnow().strftime("%Y%m%dT%H%M%S")
    n = len(data["window"])
    # disjoint from the reference sample (offset=1)
    stable_texts = reference_texts(2 * n, offset=1)
    windows = {"stable": stable_texts, "shifted": data["window"]["text"].tolist()}
    results = {}
    for name, texts in windows.items():
        cid = f"sim-{name}-{stamp}"
        out = process_comments(_records(texts, cid), cid, "demo", "simulation", "simulation")
        results[name] = {"content_id": cid, "analysis_id": out.analysis_id,
                         "drift_event_id": out.drift_event_id, "model_version": out.model_version,
                         **{k: out.drift[k] for k in ("overall_status", "drift_detected",
                                                      "severity", "reasons", "scores",
                                                      "thresholds", "sample_count")},
                         "p_value": out.drift["signals"]["embedding"]["details"]["p_value"],
                         "vocabulary": {k: out.drift["vocabulary"][k] for k in
                                        ("new_terms", "oov_rate", "unigram_jaccard")}}
        log.info("%s window: %s scores=%s", name.upper(), out.drift["overall_status"],
                 out.drift["scores"])

    rep_dir = project_path("reports", "drift")
    rep_dir.mkdir(parents=True, exist_ok=True)
    (rep_dir / f"drift_report_{stamp}.json").write_text(json.dumps(results, indent=2),
                                                        encoding="utf-8")
    md = [f"# Drift simulation report {stamp}", ""]
    for name, r in results.items():
        md += [f"## {name} window ({r['sample_count']} texts, model v{r['model_version']})",
               f"- status: **{r['overall_status']}** (drift_detected={r['drift_detected']})",
               f"- scores: {r['scores']} thresholds: {r['thresholds']}",
               f"- embedding permutation p-value: {r['p_value']}",
               f"- new terms: {', '.join(t['term'] for t in r['vocabulary']['new_terms'][:10])}",
               ""]
    (rep_dir / f"drift_report_{stamp}.md").write_text("\n".join(md), encoding="utf-8")

    stable, shifted = results["stable"], results["shifted"]
    print("\n" + "=" * 60)
    print(f"STABLE window : {stable['overall_status'].upper():8} "
          f"embedding MMD2={stable['scores']['embedding']:.4f}")
    print(f"SHIFTED window: {shifted['overall_status'].upper():8} "
          f"embedding MMD2={shifted['scores']['embedding']:.4f}")
    if shifted["drift_detected"]:
        print(f"DRIFT DETECTED  (event {shifted['drift_event_id']}: "
              f"{', '.join(shifted['reasons'])})")
    else:
        print("No drift detected on the shifted window.")
    print(f"Report: reports/drift/drift_report_{stamp}.md")
    print("=" * 60 + "\n")

    if shifted["drift_detected"] and not args.no_retrain:
        from src.pipeline.retrain import run_retraining

        out = run_retraining(shifted["drift_event_id"], scenario=args.scenario)
        print(f"Retraining decision: {out.decision} (candidate v{out.candidate_version}; "
              f"production v{out.production_before} -> v{out.production_after})")
    return 0 if shifted["drift_detected"] and not stable["drift_detected"] else 1


if __name__ == "__main__":
    sys.exit(main())
