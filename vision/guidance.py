from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from enum import Enum


# ================= HAND INFO =================
@dataclass(frozen=True)
class HandInfo:
    center: tuple[int, int] | None
    bbox: tuple[int, int, int, int] | None  # (x, y, w, h)
    area: float
    points_px: np.ndarray


# ================= EXTRACT HAND =================
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
        best_center = (
            int(sum(xs) / len(xs)),
            int(sum(ys) / len(ys)),
        )
        best_points = np.array(list(zip(xs, ys)), dtype=np.int32)

    return HandInfo(
        center=best_center,
        bbox=best_bbox,
        area=best_area,
        points_px=best_points,
    )


# ================= CONTACT =================
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

    hand_area = float((x1 - x0) * (y1 - y0))
    overlap = float(object_mask[y0:y1, x0:x1].sum())

    return overlap / max(1.0, hand_area)


# ================= CENTER GUIDANCE =================
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
    else:
        return ("up" if dy > 0 else "down"), False, distance


# ================= HAND GUIDANCE =================
def get_hand_guidance(
    hand_center: tuple[int, int] | None,
    object_center: tuple[int, int] | None,
    threshold_px: int,
    is_touching: bool = False,
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
    else:
        return ("down" if dy > 0 else "up"), None


# ================= DRAW UI =================
def draw_center_crosshair(
    frame: np.ndarray,
    frame_center: tuple[int, int],
) -> None:
    cx, cy = frame_center
    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - 18), (cx, cy + 18), (255, 255, 255), 2, cv2.LINE_AA)


# ================= STATE =================
class State(Enum):
    SEARCH = 0
    ALIGN = 1
    REACH = 2
    CONTACT = 3
    GRAB = 4


# ================= STATE MACHINE =================
class HandStateMachine:
    def __init__(self):
        self.state = State.SEARCH

        # ---------- object memory ----------
        self.last_object_center: tuple[int, int] | None = None
        self.lost_frames: int = 0
        self.max_lost_frames: int = 10       # mất hẳn sau N frame

        # ---------- occlusion (tay che vật) ----------
        # Khi tay che vật: object bị mất NHƯNG tay đang ở gần vị trí vật cũ
        self.occlusion_frames: int = 0
        self.max_occlusion_frames: int = 30  # giữ vị trí cũ lâu hơn khi bị che
        self.occlusion_proximity_threshold: int = 80  # tay trong vòng Npx = đang che

        # ---------- contact stability ----------
        self.contact_frames: int = 0
        self.contact_required: int = 3
        
        # ---------- thresholds ----------
        self.align_threshold: int = 60
        self.reach_threshold: int = 35
        self.contact_ratio_threshold: float = 0.3
        self.grab_ratio_threshold: float = 0.45
        self.contact_distance_threshold: int = 45
        self.grab_distance_threshold: int = 30

    # ------------------------------------------------------------------
    # HELPER: kiểm tra tay có đang che vật không
    # ------------------------------------------------------------------
    def _is_hand_occluding(
        self,
        hand_center: tuple[int, int] | None,
        last_object_center: tuple[int, int] | None,
    ) -> bool:
        """
        Trả về True nếu tay đang ở gần vị trí cuối cùng của vật
        → khả năng cao tay đang CHE vật, không phải vật biến mất thật sự.
        """
        if hand_center is None or last_object_center is None:
            return False

        dist = float(np.hypot(
            hand_center[0] - last_object_center[0],
            hand_center[1] - last_object_center[1],
        ))
        return dist < self.occlusion_proximity_threshold

    # ------------------------------------------------------------------
    # UPDATE
    # ------------------------------------------------------------------
    def update(
        self,
        hand_info: HandInfo,
        object_center: tuple[int, int] | None,
        contact_ratio: float,
    ) -> tuple[State, str | None]:

        # ══════════════════════════════════════════
        # BƯỚC 1 – CẬP NHẬT MEMORY VỊ TRÍ VẬT
        # ══════════════════════════════════════════
        if object_center is not None:
            # Thấy vật bình thường → cập nhật
            self.last_object_center = object_center
            self.lost_frames = 0
            self.occlusion_frames = 0
        else:
            # Không thấy vật → kiểm tra nguyên nhân
            self.lost_frames += 1

            occluded = self._is_hand_occluding(
                hand_info.center, self.last_object_center
            )

            if occluded:
                # ✅ Tay đang che vật → dùng vị trí cũ, KHÔNG reset state
                self.occlusion_frames += 1
                object_center = self.last_object_center  # tiếp tục dùng vị trí cũ

            elif (
                self.last_object_center is not None
                and self.lost_frames < self.max_lost_frames
            ):
                # ⚠️ Mất tạm thời (chưa đủ frame) → dùng vị trí cũ
                object_center = self.last_object_center

            else:
                # ❌ Mất thật sự → reset về SEARCH
                self.state = State.SEARCH
                self.last_object_center = None
                self.occlusion_frames = 0
                return self.state, "searching"

        # ══════════════════════════════════════════
        # BƯỚC 2 – TÍNH KHOẢNG CÁCH TAY ↔ VẬT
        # ══════════════════════════════════════════
        if object_center is None:
            return self.state, "searching"

        hx, hy = hand_info.center if hand_info.center else (None, None)
        ox, oy = object_center

        distance: float = 9999.0
        if hx is not None:
            distance = float(np.hypot(hx - ox, hy - oy))

        # ══════════════════════════════════════════
        # BƯỚC 3 – STATE TRANSITIONS
        # ══════════════════════════════════════════

        # ── SEARCH ──────────────────────────────
        if self.state == State.SEARCH:
            self.state = State.ALIGN
            return self.state, "target found"

        # ── ALIGN ───────────────────────────────
        if self.state == State.ALIGN:
            if hand_info.center is None:
                return self.state, "show hand"

            direction, hint = get_hand_guidance(
                hand_info.center,
                object_center,
                threshold_px=self.align_threshold,
            )

            if distance < self.align_threshold:
                self.state = State.REACH

            return self.state, direction

        # ── REACH ───────────────────────────────
        if self.state == State.REACH:
            direction, hint = get_hand_guidance(
                hand_info.center,
                object_center,
                threshold_px=self.reach_threshold,
            )

            is_close = distance < self.contact_distance_threshold
            is_overlap = contact_ratio > self.contact_ratio_threshold

            # Tay che vật thì coi như đã có tiếp xúc → tính contact_frames luôn
            is_occluding = self.occlusion_frames > 0

            if is_close and (is_overlap or is_occluding):
                self.contact_frames += 1
            else:
                self.contact_frames = 0

            if self.contact_frames >= self.contact_required:
                self.state = State.CONTACT

            return self.state, direction or "move forward"

        # ── CONTACT ─────────────────────────────
        if self.state == State.CONTACT:
            is_occluding = self.occlusion_frames > 0

            # Nếu tay đang che vật → tiếp tục đếm hướng đến GRAB
            if (
                (contact_ratio > self.grab_ratio_threshold or is_occluding)
                and distance < self.grab_distance_threshold
            ):
                self.state = State.GRAB

            return self.state, "contact – hold still"

        # ── GRAB ────────────────────────────────
        if self.state == State.GRAB:
            return self.state, "grab confirmed ✓"

        return self.state, None
