#!/usr/bin/env python3
"""sam3_viz.py - Regenerate --viz overlay videos straight from existing sam3_out
sidecars, without re-running SAM3. Use after a fix to write_viz()/interpolate_box(),
or after sam3_reval.py updates eval/bowler_person_match in place.
"""
import argparse
import json
import pathlib

from sam3_tracks import write_viz


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video_dir", type=pathlib.Path, default=pathlib.Path("videos"))
    ap.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("sam3_out"))
    ap.add_argument("--viz_dir", type=pathlib.Path, default=None, help="Defaults to <out_dir>/viz")
    ap.add_argument("--videos", default="",
                    help="Comma-separated video stems to render (e.g. IMG_8538,IMG_8352); "
                         "default renders every sidecar in --out_dir.")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    viz_dir = args.viz_dir or (args.out_dir / "viz")
    viz_dir.mkdir(parents=True, exist_ok=True)

    sidecars = sorted(args.out_dir.glob("*.json"))
    wanted = {s.strip() for s in args.videos.split(",") if s.strip()}
    if wanted:
        sidecars = [p for p in sidecars if p.stem in wanted]
    if args.limit:
        sidecars = sidecars[:args.limit]

    n_ok, n_skip = 0, 0
    for i, sc_path in enumerate(sidecars, 1):
        sidecar = json.load(open(sc_path, encoding="utf-8"))
        video_name = sidecar.get("video")
        video_path = args.video_dir / video_name if video_name else None
        if not video_path or not video_path.exists():
            print(f"[{i}/{len(sidecars)}] skip {sc_path.name}: video not found at {video_path}")
            n_skip += 1
            continue
        out_path = viz_dir / f"{sc_path.stem}.mp4"
        print(f"[{i}/{len(sidecars)}] {sc_path.stem} -> {out_path}")
        write_viz(video_path, sidecar, out_path)
        n_ok += 1

    print(f"\nwrote {n_ok} viz videos, skipped {n_skip} -> {viz_dir}")


if __name__ == "__main__":
    main()
