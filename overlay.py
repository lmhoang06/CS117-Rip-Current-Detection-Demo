"""Polygon mask overlays for Ultralytics YOLO-seg results (from ripvis_vis_pred.py)."""

from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def iter_result_polygons(r0) -> Iterable[np.ndarray]:
    """
    Yield polygons as (N, 1, 2) int32 arrays for cv2.fillPoly.
    Ultralytics seg polygons are available as r0.masks.xy in pixel coords.
    """
    if r0 is None or getattr(r0, "masks", None) is None:
        return
    polys = getattr(r0.masks, "xy", None)
    if polys is None:
        return
    for poly in polys:
        pts = np.asarray(poly, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        pts = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        yield pts


def draw_polygons_overlay(
    bgr: np.ndarray,
    polygons: Iterable[np.ndarray],
    *,
    alpha: float,
    color_bgr: tuple[int, int, int],
) -> np.ndarray:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0,1], got {alpha}")

    overlay = bgr.copy()
    any_poly = False
    for pts in polygons:
        any_poly = True
        cv2.fillPoly(overlay, [pts], color_bgr)

    if not any_poly:
        return bgr
    return cv2.addWeighted(overlay, alpha, bgr, 1.0 - alpha, 0.0)
