#!/usr/bin/env python3
"""sam3_reval.py - Recompute eval in existing sam3_out sidecars without re-running SAM3
(e.g. after a fix to evaluate()/interpolate_box()). Reuses each sidecar's own
person_tracks plus its original videos/*.json annotation, and overwrites the sidecar
in place. Does NOT recompute bowler_person_match: the box-prompt bowler track is lower
quality than the person track it'd be matched against, so that match isn't trustworthy.
"""
import argparse
import json
import pathlib

from sam3_tracks import evaluate, load_annotation


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("sam3_out"))
    ap.add_argument("--dry_run", action="store_true", help="Report stats without writing.")
    args = ap.parse_args()

    n_updated, n_skipped = 0, 0
    before, after = [], []

    for sc_path in sorted(args.out_dir.glob("*.json")):
        sidecar = json.load(open(sc_path, encoding="utf-8"))
        video_name = sidecar.get("video")
        anno_path = args.video_dir / (pathlib.Path(video_name).stem + ".json") if video_name else None
        anno = load_annotation(anno_path) if anno_path and anno_path.exists() else None
        if anno is None:
            print(f"skip {sc_path.name}: no matching annotation/video at {anno_path}")
            n_skipped += 1
            continue

        before.extend(row["best_person_iou"] for row in sidecar.get("eval", {}).get("per_event", []))

        if "person_tracks" not in sidecar:
            n_skipped += 1
            continue
        sidecar["eval"] = evaluate(anno, sidecar["person_tracks"])
        after.extend(row["best_person_iou"] for row in sidecar["eval"]["per_event"])

        if not args.dry_run:
            with open(sc_path, "w", encoding="utf-8") as f:
                json.dump(sidecar, f, indent=1)
        n_updated += 1

    def q(xs, p):
        xs = sorted(xs)
        return xs[int(p * (len(xs) - 1))] if xs else float("nan")

    print(f"\n{'would update' if args.dry_run else 'updated'} {n_updated} sidecars, skipped {n_skipped}")
    print(f"best_person_iou median: before={q(before, .5):.3f} after={q(after, .5):.3f}")
    for t in (0.3, 0.5, 0.7):
        print(f"  events IoU>{t}: before={sum(v > t for v in before)}/{len(before)}"
              f"  after={sum(v > t for v in after)}/{len(after)}")


if __name__ == "__main__":
    main()
