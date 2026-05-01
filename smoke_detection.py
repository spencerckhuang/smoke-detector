"""
smoke_detection.py — Deterministic CV-based smoke score for circular tissue samples.

Pipeline:
    1. Detect the inner circle of the tissue via Hough Circle Transform.
    2. Mask everything outside the circle.
    3. Compute three smoke indicators inside the mask:
         - Mean HSV saturation       (smoke desaturates → lower = smokier)
         - Canny edge density        (smoke blurs detail → lower = smokier)
         - Mean dark channel prior   (smoke adds airlight → higher = smokier)
    4. Normalize each to [0, 1] against calibrated "clear" / "smoky" anchors.
    5. Combine with weights into a 0–100% smoke score.

The clear/smoky anchor values are dataset-dependent. Calibrate them on a handful
of known-clear and known-smoky tissues before using the score for anything real.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SmokeMetrics:
    """Raw measurements, per-feature normalized scores, and final weighted score."""
    # Raw measurements
    saturation_mean: float        # 0–255, mean HSV S inside circle
    edge_density: float           # 0–1, fraction of in-circle pixels that are Canny edges
    dark_channel_mean: float      # 0–255, mean dark channel prior inside circle

    # Per-feature normalized smoke contributions in [0, 1] (1 = maximally smoky)
    saturation_score: float
    edge_score: float
    dark_channel_score: float

    # Final score in [0, 100]
    smoke_score: float


# ---------------------------------------------------------------------------
# Circle detection & masking
# ---------------------------------------------------------------------------

def detect_inner_circle(
    image: np.ndarray,
    min_radius_frac: float = 0.10,
    max_radius_frac: float = 0.50,
    dp: float = 1.2,
    param1: float = 100.0,
    param2: float = 30.0,
) -> Optional[Tuple[int, int, int]]:
    """
    Detect the inner circle of a tissue sample with Hough Circle Transform.

    Radius bounds are expressed as fractions of min(height, width) so the same
    parameters work across image resolutions.

    Args:
        image:            BGR image as a numpy array.
        min_radius_frac:  Minimum circle radius / min(H, W).
        max_radius_frac:  Maximum circle radius / min(H, W).
        dp:               Inverse ratio of accumulator resolution.
        param1:           Upper Canny threshold used inside HoughCircles.
        param2:           Accumulator threshold; lower → more (and noisier) circles.

    Returns:
        (cx, cy, r) of the smallest detected circle, or None if no circle is found.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    h, w = gray.shape
    min_dim = min(h, w)
    min_radius = int(min_dim * min_radius_frac)
    max_radius = int(min_dim * max_radius_frac)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dim,  # we expect a single dominant circle
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is None:
        return None

    # circles has shape (1, n, 3); pick the smallest-radius one (the inner circle).
    circles = np.round(circles[0]).astype(int)
    cx, cy, r = sorted(circles, key=lambda c: c[2])[0]
    return int(cx), int(cy), int(r)


def make_circle_mask(
    shape: Tuple[int, ...],
    center: Tuple[int, int],
    radius: int,
    shrink: float = 0.95,
) -> np.ndarray:
    """
    Build a uint8 binary mask for a circle.

    `shrink` slightly contracts the radius to avoid sampling the boundary
    artifacts (well rim, lighting falloff, etc.) that often live right at
    the detected edge.
    """
    mask = np.zeros(shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, int(radius * shrink), 255, thickness=-1)
    return mask


# ---------------------------------------------------------------------------
# Per-feature measurements
# ---------------------------------------------------------------------------

def compute_saturation(image: np.ndarray, mask: np.ndarray) -> float:
    """Mean HSV saturation inside the mask. Lower is smokier."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return float(cv2.mean(saturation, mask=mask)[0])


def compute_edge_density(
    image: np.ndarray,
    mask: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
) -> float:
    """Fraction of in-circle pixels that are Canny edges. Lower is smokier."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)
    edges_in_mask = cv2.bitwise_and(edges, edges, mask=mask)

    n_edge = int(np.count_nonzero(edges_in_mask))
    n_mask = int(np.count_nonzero(mask))
    if n_mask == 0:
        return 0.0
    return n_edge / n_mask


def compute_dark_channel(
    image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = 15,
) -> float:
    """
    Mean dark channel prior (He et al., 2009) inside the mask. Higher is smokier.

    Dark channel = min across BGR channels, then min in a local square patch.
    Haze/smoke adds airlight to all channels, raising the per-pixel min, so the
    dark channel of smoky regions is noticeably higher than that of clear ones.
    """
    min_rgb = np.min(image, axis=2).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_rgb, kernel)  # erosion of a grayscale image == local min
    return float(cv2.mean(dark, mask=mask)[0])


# ---------------------------------------------------------------------------
# Top-level scoring
# ---------------------------------------------------------------------------

def _normalize(value: float, clear: float, smoky: float) -> float:
    """Linearly map `value` from the [clear, smoky] range onto [0, 1], clipped."""
    if clear == smoky:
        return 0.0
    return float(np.clip((value - clear) / (smoky - clear), 0.0, 1.0))


def analyze_smoke(
    image: np.ndarray,
    *,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    saturation_clear: float = 120.0,
    saturation_smoky: float = 20.0,
    edge_clear: float = 0.10,
    edge_smoky: float = 0.01,
    dark_clear: float = 20.0,
    dark_smoky: float = 150.0,
    circle: Optional[Tuple[int, int, int]] = None,
) -> Tuple[SmokeMetrics, Tuple[int, int, int]]:
    """
    Compute a 0–100 smoke score for a tissue sample image.

    Args:
        image:               BGR image (numpy array).
        weights:             (saturation, edge, dark_channel). Should sum to 1.
        saturation_clear/_smoky: Anchor mean-saturation values for normalization.
        edge_clear/_smoky:       Anchor edge-density values for normalization.
        dark_clear/_smoky:       Anchor dark-channel values for normalization.
        circle:              Optional pre-computed (cx, cy, r). If None, detected
                             automatically with default Hough parameters.

    Returns:
        (SmokeMetrics, (cx, cy, r)) — the metrics and the circle that was used.

    Raises:
        ValueError: If no inner circle can be detected.
    """
    if circle is None:
        circle = detect_inner_circle(image)
        if circle is None:
            raise ValueError("Could not detect an inner circle in the image.")

    cx, cy, r = circle
    mask = make_circle_mask(image.shape, (cx, cy), r)

    sat = compute_saturation(image, mask)
    edge = compute_edge_density(image, mask)
    dark = compute_dark_channel(image, mask)

    # Note the direction: saturation/edge are inverted (clear > smoky in raw value),
    # so we swap the args to _normalize so that 1.0 always means "maximally smoky".
    sat_score = _normalize(sat, clear=saturation_clear, smoky=saturation_smoky)
    edge_score = _normalize(edge, clear=edge_clear, smoky=edge_smoky)
    dark_score = _normalize(dark, clear=dark_clear, smoky=dark_smoky)

    w_sat, w_edge, w_dark = weights
    weighted = w_sat * sat_score + w_edge * edge_score + w_dark * dark_score
    smoke_score = float(np.clip(weighted * 100.0, 0.0, 100.0))

    metrics = SmokeMetrics(
        saturation_mean=sat,
        edge_density=edge,
        dark_channel_mean=dark,
        saturation_score=sat_score,
        edge_score=edge_score,
        dark_channel_score=dark_score,
        smoke_score=smoke_score,
    )
    return metrics, (cx, cy, r)


# ---------------------------------------------------------------------------
# Visualization helpers (handy while tuning anchors / weights)
# ---------------------------------------------------------------------------

def visualize(
    image: np.ndarray,
    metrics: SmokeMetrics,
    circle: Tuple[int, int, int],
) -> np.ndarray:
    """Overlay the detected circle and metric values on a copy of the image."""
    out = image.copy()
    cx, cy, r = circle
    cv2.circle(out, (cx, cy), r, (0, 255, 0), 2)
    cv2.circle(out, (cx, cy), 3, (0, 0, 255), -1)

    lines = [
        f"Smoke: {metrics.smoke_score:5.1f}%",
        f"Sat:   {metrics.saturation_mean:6.1f}  -> {metrics.saturation_score:.2f}",
        f"Edge:  {metrics.edge_density:6.4f}  -> {metrics.edge_score:.2f}",
        f"Dark:  {metrics.dark_channel_mean:6.1f}  -> {metrics.dark_channel_score:.2f}",
    ]
    for i, line in enumerate(lines):
        y = 30 + i * 28
        # Black halo for legibility on light backgrounds.
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score a tissue image for smokiness.")
    parser.add_argument("image", help="Path to BGR-readable image.")
    parser.add_argument("--debug-out", default=None,
                        help="If set, write an annotated debug image here.")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")

    metrics, circle = analyze_smoke(img)
    print(f"Smoke score: {metrics.smoke_score:.1f}%")
    print(f"  saturation:   raw={metrics.saturation_mean:7.2f}   score={metrics.saturation_score:.3f}")
    print(f"  edge density: raw={metrics.edge_density:7.4f}   score={metrics.edge_score:.3f}")
    print(f"  dark channel: raw={metrics.dark_channel_mean:7.2f}   score={metrics.dark_channel_score:.3f}")

    if args.debug_out:
        cv2.imwrite(args.debug_out, visualize(img, metrics, circle))
        print(f"Wrote debug image: {args.debug_out}")