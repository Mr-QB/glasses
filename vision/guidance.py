from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np


@dataclass(frozen=True)
class HandInfo:
    center: tuple[int, int] | None
    bbox: tuple[int, int, int, int] | None
    area: float
    points_px: np.ndarray


def extract_primary_hand(
    hand_points: Sequence[Sequence[tuple[float, float]]],
    frame_width: int,
    frame_height: int,
) -> HandInfo:
    best_bbox = None
    best_points = np.empty((0, 2), dtype=np.int32)
    best_area = 0.0
    best_center = None

    for hand in hand_points:
        if not hand:
            continue
        xs = [int(x * frame_width) for x, _ in hand]
        ys = [int(y * frame_height) for _, y in hand]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        area = float(w * h)
        if area <= best_area:
            continue
        best_area = area
        best_bbox = (x0, y0, w, h)
        best_center = (int(sum(xs) / len(xs)), int(sum(ys) / len(ys)))
        best_points = np.array(list(zip(xs, ys)), dtype=np.int32)

    return HandInfo(
        center=best_center,
        bbox=best_bbox,
        area=best_area,
        points_px=best_points,
    )


def estimate_contact_ratio(
    hand_bbox: tuple[int, int, int, int] | None,
    object_mask: np.ndarray | None,
) -> float:
    if hand_bbox is None or object_mask is None or object_mask.size == 0:
        return 0.0

    mask_h, mask_w = object_mask.shape[:2]
    x, y, w, h = hand_bbox
    x0 = max(0, x)
    y0 = max(0, y)
    x1 = min(mask_w, x + w)
    y1 = min(mask_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return 0.0

    hand_roi_area = float((x1 - x0) * (y1 - y0))
    overlap = float(object_mask[y0:y1, x0:x1].sum())
    return overlap / max(1.0, hand_roi_area)


def get_center_guidance(
    object_center: tuple[int, int] | None,
    frame_center: tuple[int, int],
    threshold_px: int,
) -> tuple[str | None, bool, float]:
    if object_center is None:
        return None, False, 0.0

    ox, oy = object_center
    cx, cy = frame_center
    dx = cx - ox
    dy = cy - oy
    distance = float(np.hypot(dx, dy))
    if distance <= threshold_px:
        return "centered", True, distance

    if abs(dx) >= abs(dy):
        return ("left" if dx > 0 else "right"), False, distance
    return ("up" if dy > 0 else "down"), False, distance


def get_hand_guidance(
    hand_center: tuple[int, int] | None,
    object_center: tuple[int, int] | None,
    threshold_px: int,
    is_touching: bool,
) -> tuple[str | None, str | None]:
    if hand_center is None or object_center is None:
        return None, "show hand in frame"
    if is_touching:
        return "forward", "contact detected"

    hx, hy = hand_center
    ox, oy = object_center
    dx = ox - hx
    dy = oy - hy
    if abs(dx) <= threshold_px and abs(dy) <= threshold_px:
        return "forward", "reach forward"

    if abs(dx) >= abs(dy):
        return ("right" if dx > 0 else "left"), None
    return ("down" if dy > 0 else "up"), None


def draw_center_crosshair(frame: np.ndarray, frame_center: tuple[int, int]) -> None:
    cx, cy = frame_center
    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 18), (cx, cy + 18), (255, 255, 255), 2, cv2.LINE_AA)
