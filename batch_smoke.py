"""
Run smoke_detection.analyze_smoke over a dataset and save per-frame results to CSV.

Layout assumed: <dataset_root>/<tissue_id>/<phase>/<frame>.{png,jpg,jpeg,...}
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterator, Tuple

import cv2

from smoke_detection import analyze_smoke


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

FIELDNAMES = [
    "tissue", "seq", "phase", "frame_path",
    "smoke_score",
    "saturation_mean", "edge_density", "dark_channel_mean",
    "saturation_score", "edge_score", "dark_channel_score",
    "status",
]


def iter_frames(dataset_root: Path, phase: str) -> Iterator[Tuple[str, str, Path]]:
    """Yield (tissue, seq, frame_path) for every image under <root>/<phase>/<tissue>/<seq>/."""
    phase_dir = dataset_root / phase
    if not phase_dir.is_dir():
        raise FileNotFoundError(f"Phase directory not found: {phase_dir}")
    for tissue_dir in sorted(p for p in phase_dir.iterdir() if p.is_dir()):
        for seq_dir in sorted(p for p in tissue_dir.iterdir() if p.is_dir()):
            for frame_path in sorted(seq_dir.iterdir()):
                if frame_path.is_file() and frame_path.suffix.lower() in IMAGE_EXTS:
                    yield tissue_dir.name, seq_dir.name, frame_path

def process_dataset(dataset_root: Path, phase: str, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_fail = 0

    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()

        for tissue, seq, frame_path in iter_frames(dataset_root, phase):
            row = {k: "" for k in FIELDNAMES}
            row.update(tissue=tissue, seq=seq, phase=phase, frame_path=str(frame_path))

            img = cv2.imread(str(frame_path))
            if img is None:
                row["status"] = "read_failed"
                n_fail += 1
            else:
                try:
                    metrics, _ = analyze_smoke(img)
                    row.update(
                        smoke_score=metrics.smoke_score,
                        saturation_mean=metrics.saturation_mean,
                        edge_density=metrics.edge_density,
                        dark_channel_mean=metrics.dark_channel_mean,
                        saturation_score=metrics.saturation_score,
                        edge_score=metrics.edge_score,
                        dark_channel_score=metrics.dark_channel_score,
                        status="ok",
                    )
                    n_ok += 1
                except ValueError as e:
                    row["status"] = f"analyze_failed: {e}"
                    n_fail += 1

            writer.writerow(row)
            total = n_ok + n_fail
            if total % 200 == 0:
                f.flush()
                print(f"  {total} frames processed ({n_fail} failed)")

    print(f"Done. {n_ok} ok, {n_fail} failed. Wrote {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--phase", default="3_resect")
    parser.add_argument("--output", type=Path, default=Path("smoke_scores.csv"))
    args = parser.parse_args()

    process_dataset(args.dataset_root, args.phase, args.output)