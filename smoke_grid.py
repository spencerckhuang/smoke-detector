"""
Render a deciles x N-columns grid of frames sampled by smoke score, from the CSV
produced by batch_smoke.py.

Within each decile bin, frames are sorted by smoke_score and N evenly-spaced
positional samples are taken (quantile-style — natural skew toward dense regions).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def pick_frames_for_bin(df_bin: pd.DataFrame, n_cols: int) -> pd.DataFrame:
    """Sort by score, return n_cols rows at evenly-spaced positional indices."""
    if df_bin.empty:
        return df_bin
    sorted_bin = (
        df_bin.sort_values(["smoke_score", "frame_path"])  # frame_path = stable tiebreak
        .reset_index(drop=True)
    )
    if len(sorted_bin) <= n_cols:
        return sorted_bin
    idx = np.linspace(0, len(sorted_bin) - 1, n_cols).round().astype(int)
    return sorted_bin.iloc[idx].reset_index(drop=True)


def make_grid(
    csv_path: Path,
    output_path: Path,
    n_cols: int = 8,
    n_bins: int = 10,
    title: str = "Smoke score ladder",
) -> None:
    df = pd.read_csv(csv_path)
    df = df[df["status"] == "ok"].copy()
    if df.empty:
        raise SystemExit("No successful rows in CSV.")

    edges = np.linspace(0, 100, n_bins + 1)
    fig, axes = plt.subplots(n_bins, n_cols, figsize=(n_cols * 1.8, n_bins * 1.8))
    fig.suptitle(title, fontsize=12)

    for row_i in range(n_bins):
        lo, hi = edges[row_i], edges[row_i + 1]
        # last bin inclusive on the right; otherwise [lo, hi)
        if row_i == n_bins - 1:
            df_bin = df[(df["smoke_score"] >= lo) & (df["smoke_score"] <= hi)]
        else:
            df_bin = df[(df["smoke_score"] >= lo) & (df["smoke_score"] < hi)]

        picks = pick_frames_for_bin(df_bin, n_cols)

        axes[row_i, 0].set_ylabel(
            f"{int(lo)}-{int(hi)}%\n(n={len(df_bin):,})",
            rotation=90, fontsize=9, labelpad=10,
        )

        for col_i in range(n_cols):
            ax = axes[row_i, col_i]
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)

            if col_i >= len(picks):
                continue
            r = picks.iloc[col_i]
            img = cv2.imread(r["frame_path"])
            if img is not None:
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax.set_title(
                f"{int(round(r['smoke_score']))}%\n{r['tissue']}\n{r['phase']}",
                fontsize=7,
            )

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--output", type=Path, default=Path("smoke_grid.png"))
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--title", default="Smoke score ladder")
    args = parser.parse_args()

    make_grid(args.csv_path, args.output, args.cols, args.bins, args.title)