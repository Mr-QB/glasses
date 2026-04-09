from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from threading import RLock
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import torch
from ultralytics import YOLOE

from vision.guidance import (
    draw_center_crosshair,
    estimate_contact_ratio,
    extract_primary_hand,
    get_center_guidance,
    get_hand_guidance,
)
from vision.hand_pose import HandPoseProcessor
from vision.item_search_state import ItemSearchState


@dataclass
class DetectedObject:
    mask: np.ndarray
    contour: np.ndarray
    center: tuple[int, int]
    area: float


class FrameProcessor:
    _GUIDANCE_IDS = {"left", "right", "up", "down", "forward", "none"}

    def __init__(
        self,
        model_path: str = "yolov8m-seg.pt",
        device: str = "auto",
        use_fp16: bool = True,
        infer_imgsz: int = 416,
        conf_threshold: float = 0.4,
        iou_threshold: float = 0.6,
        max_det: int = 50,
        yoloe_prompts: tuple[str, ...] | list[str] | str = ("glass",),
        enable_hand_pose: bool = True,
        max_num_hands: int = 2,
        hand_detection_confidence: float = 0.5,
        hand_tracking_confidence: float = 0.5,
        enable_item_search: bool = True,
        lock_required_frames: int = 8,
        center_threshold_px: int = 35,
        center_hold_frames: int = 6,
        hand_guidance_threshold_px: int = 35,
        track_lost_tolerance_frames: int = 10,
        contact_overlap_threshold: float = 0.12,
        occlusion_distance_threshold_px: int = 80,
        occlusion_overlap_threshold: float = 0.12,
        occlusion_stable_frames: int = 2,
        occlusion_reacquire_wait_frames: int = 3,
        occlusion_timeout_seconds: float = 3.0,
        forward_resume_hold_frames: int = 2,
        on_guidance_change: Callable[[str], None] | None = None,
    ) -> None:
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        if self.use_fp16:
            # YOLOE-seg can crash with Half/Float mismatch in postprocess.
            self.use_fp16 = False
            print("[YOLOE] fp16 disabled due to segmentation dtype mismatch bug.")
        self.infer_imgsz = infer_imgsz
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_det = max_det
        self.yoloe_prompts = self._normalize_prompts(yoloe_prompts)
        self._prompt_lock = RLock()
        self.hand_pose: HandPoseProcessor | None = None
        self._on_guidance_change = on_guidance_change

        self.enable_item_search = enable_item_search
        self.lock_required_frames = max(1, lock_required_frames)
        self.center_threshold_px = max(1, center_threshold_px)
        self.center_hold_frames = max(1, center_hold_frames)
        self.hand_guidance_threshold_px = max(1, hand_guidance_threshold_px)
        self.track_lost_tolerance_frames = max(1, track_lost_tolerance_frames)
        self.contact_overlap_threshold = max(0.0, min(1.0, contact_overlap_threshold))
        self.occlusion_distance_threshold_px = max(1, occlusion_distance_threshold_px)
        self.occlusion_overlap_threshold = max(
            0.0, min(1.0, occlusion_overlap_threshold)
        )
        self.occlusion_stable_frames = max(1, occlusion_stable_frames)
        self.occlusion_reacquire_wait_frames = max(1, occlusion_reacquire_wait_frames)
        self.occlusion_timeout_seconds = max(0.1, float(occlusion_timeout_seconds))
        self.forward_resume_hold_frames = max(0, forward_resume_hold_frames)

        self.state = ItemSearchState.SEGMENT
        self._stable_detection_frames = 0
        self._centered_frames = 0
        self._lost_frames = 0
        self._last_object_center: tuple[int, int] | None = None
        self._last_visible_object_center: tuple[int, int] | None = None
        self._last_hand_center: tuple[int, int] | None = None
        self._last_contact_ratio = 0.0
        self._overlap_evidence_frames = 0
        self._occlusion_overlap_memory_frames = max(2, self.occlusion_stable_frames * 3)
        self._forward_evidence_frames = 0
        self._occlusion_candidate_frames = 0
        self._occlusion_active = False
        self._occlusion_active_since: float | None = None
        self._occlusion_reference_center: tuple[int, int] | None = None
        self._resume_forward_frames = 0
        self._last_command = "none"
        self._last_update_ts: float | None = None
        self._fps_ema = 0.0

        self._load_model()
        print(f"YOLOE Segmentation device: {self.device} | fp16: {self.use_fp16}")

        if enable_hand_pose:
            self.hand_pose = HandPoseProcessor(
                max_num_hands=max_num_hands,
                min_detection_confidence=hand_detection_confidence,
                min_tracking_confidence=hand_tracking_confidence,
            )
            print(f"[MP] {self.hand_pose.get_status()}")
        else:
            print("[MP] Hand tracking disabled by settings.")

        self._warmup()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _normalize_prompts(
        yoloe_prompts: tuple[str, ...] | list[str] | str,
    ) -> tuple[str, ...]:
        if isinstance(yoloe_prompts, str):
            prompts: Iterable[str] = [yoloe_prompts]
        else:
            prompts = yoloe_prompts
        return tuple(p.strip() for p in prompts if isinstance(p, str) and p.strip())

    def _warmup(self) -> None:
        warmup_frame = np.zeros((self.infer_imgsz, self.infer_imgsz, 3), dtype=np.uint8)
        print("[YOLOE] Warming up model...")
        self.model(
            warmup_frame,
            verbose=False,
            device=self.device,
            half=self.use_fp16,
            imgsz=self.infer_imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
        )
        print("[YOLOE] Warmup done.")

    def _load_model(self) -> None:
        prompt_text = ", ".join(self.yoloe_prompts) if self.yoloe_prompts else "none"
        print(
            f"[YOLOE] Reloading model from {self.model_path} with prompts: {prompt_text}"
        )
        self.model = YOLOE(self.model_path, task="segment")
        self.model.to(self.device)
        self._apply_open_vocab_prompts()
        self._warmup()
        print(f"[YOLOE] Model reload complete for prompts: {prompt_text}")

    def _apply_open_vocab_prompts(self) -> None:
        if not self.yoloe_prompts:
            print("[YOLOE] No prompt provided. Running as closed-set model.")
            return
        try:
            if hasattr(self.model, "set_classes"):
                self.model.set_classes(list(self.yoloe_prompts))
            elif hasattr(self.model, "set_vocab"):
                self.model.set_vocab(list(self.yoloe_prompts))
            else:
                raise AttributeError("model has neither set_classes nor set_vocab")
            print(f"[YOLOE] Open-vocab prompts: {', '.join(self.yoloe_prompts)}")
        except Exception as exc:
            print(
                f"[YOLOE] Cannot apply prompts ({type(exc).__name__}: {exc!r}). Running without open-vocab prompts."
            )

    def set_prompts(self, prompts: tuple[str, ...] | list[str] | str) -> None:
        normalized_prompts = self._normalize_prompts(prompts)
        with self._prompt_lock:
            if normalized_prompts == self.yoloe_prompts:
                print(
                    f"[YOLOE] Prompt unchanged, keeping current label: "
                    f"{', '.join(self.yoloe_prompts) if self.yoloe_prompts else 'none'}"
                )
                return

            old_prompt_text = (
                ", ".join(self.yoloe_prompts) if self.yoloe_prompts else "none"
            )
            new_prompt_text = (
                ", ".join(normalized_prompts) if normalized_prompts else "none"
            )
            print(
                f"[YOLOE] Prompt update requested: {old_prompt_text} -> {new_prompt_text}"
            )
            self.yoloe_prompts = normalized_prompts
            self._load_model()
            self._set_state(ItemSearchState.SEGMENT)
            self._lost_frames = 0
            self._last_object_center = None
            self._last_visible_object_center = None
            self._last_hand_center = None
            self._last_contact_ratio = 0.0
            self._overlap_evidence_frames = 0
            self._forward_evidence_frames = 0
            self._occlusion_candidate_frames = 0
            self._occlusion_active = False
            self._occlusion_active_since = None
            self._occlusion_reference_center = None
            self._resume_forward_frames = 0
            self._last_command = "none"
            self._last_update_ts = None
            self._fps_ema = 0.0
            print(
                f"[YOLOE] Prompt switched successfully: {old_prompt_text} -> {new_prompt_text}"
            )

    def _extract_masks(
        self, results: list[Any], frame_shape: tuple[int, int]
    ) -> list[np.ndarray]:
        masks: list[np.ndarray] = []
        if not results:
            return masks

        r0 = results[0]
        if getattr(r0, "masks", None) is None:
            return masks

        mask_data = getattr(r0.masks, "data", None)
        if mask_data is None:
            return masks

        mask_h, mask_w = frame_shape
        conf_values = None
        if (
            getattr(r0, "boxes", None) is not None
            and getattr(r0.boxes, "conf", None) is not None
        ):
            conf_values = r0.boxes.conf

        for idx, mask_tensor in enumerate(mask_data):
            conf = (
                float(conf_values[idx].item())
                if conf_values is not None and idx < len(conf_values)
                else 1.0
            )
            if conf < self.conf_threshold:
                continue
            mask_np = mask_tensor.detach().cpu().numpy()
            if mask_np.shape[:2] != (mask_h, mask_w):
                mask_np = cv2.resize(
                    mask_np, (mask_w, mask_h), interpolation=cv2.INTER_LINEAR
                )
            mask_bin = (mask_np > 0.5).astype(np.uint8)
            if int(mask_bin.sum()) > 0:
                masks.append(mask_bin)
        return masks

    @staticmethod
    def _build_detected_object(mask_bin: np.ndarray) -> DetectedObject | None:
        contours, _ = cv2.findContours(
            mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        if not contours:
            return None
        contour = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(contour))
        if area <= 0.0:
            return None
        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            center = tuple(np.mean(contour.reshape(-1, 2), axis=0).astype(int))
        else:
            center = (
                int(moments["m10"] / moments["m00"]),
                int(moments["m01"] / moments["m00"]),
            )
        return DetectedObject(
            mask=mask_bin,
            contour=contour.reshape(-1, 2),
            center=center,
            area=area,
        )

    def _select_target(self, masks: list[np.ndarray]) -> DetectedObject | None:
        if not masks:
            return None

        objects: list[DetectedObject] = []
        for mask in masks:
            obj = self._build_detected_object(mask)
            if obj is not None:
                objects.append(obj)
        if not objects:
            return None

        if self._last_object_center is None:
            return max(objects, key=lambda obj: obj.area)

        last = np.array(self._last_object_center, dtype=np.float32)
        return max(
            objects,
            key=lambda obj: obj.area
            - 0.25
            * float(np.linalg.norm(np.array(obj.center, dtype=np.float32) - last)),
        )

    def _reset_tracking_state(self) -> None:
        self._stable_detection_frames = 0
        self._centered_frames = 0

    def _set_state(self, state: ItemSearchState) -> None:
        if self.state == state:
            return
        self.state = state
        self._reset_tracking_state()

    def _normalize_guidance_id(self, value: str | None) -> str:
        if value is None:
            return "none"
        normalized = str(value).strip().lower()
        if normalized in self._GUIDANCE_IDS:
            return normalized
        return "none"

    def _clear_occlusion(self) -> None:
        self._occlusion_candidate_frames = 0
        self._occlusion_active = False
        self._occlusion_active_since = None
        self._occlusion_reference_center = None
        self._resume_forward_frames = 0
        self._overlap_evidence_frames = 0
        self._forward_evidence_frames = 0

    def _return_guidance(self, value: str | None) -> tuple[str, str]:
        normalized = self._normalize_guidance_id(value)
        previous = self._last_command
        self._last_command = normalized
        if (
            normalized != previous
            and self._on_guidance_change is not None
        ):
            try:
                self._on_guidance_change(normalized)
            except Exception as exc:
                print(f"[GUIDE] on_guidance_change callback failed: {exc}")
        if normalized == "forward":
            base_frames = max(2, self.occlusion_stable_frames + 1)
            if 0.0 < self._fps_ema <= 6.0:
                base_frames = max(base_frames, 4)
            self._forward_evidence_frames = base_frames
        return normalized, normalized

    def _update_fps_estimate(self, now: float) -> None:
        if self._last_update_ts is None:
            self._last_update_ts = now
            return
        dt = now - self._last_update_ts
        self._last_update_ts = now
        if dt <= 1e-6:
            return
        instant_fps = 1.0 / dt
        if self._fps_ema <= 0.0:
            self._fps_ema = instant_fps
        else:
            self._fps_ema = self._fps_ema * 0.85 + instant_fps * 0.15

    def _effective_occlusion_stable_frames(self) -> int:
        if 0.0 < self._fps_ema <= 6.0:
            return 1
        return self.occlusion_stable_frames

    def _effective_occlusion_wait_frames(self) -> int:
        if 0.0 < self._fps_ema <= 6.0:
            return max(1, self.occlusion_reacquire_wait_frames - 1)
        return self.occlusion_reacquire_wait_frames

    def _enter_reacquire(self) -> tuple[str, str]:
        self._set_state(ItemSearchState.SEGMENT)
        self._stable_detection_frames = 0
        self._centered_frames = 0
        self._lost_frames = 0
        self._clear_occlusion()
        return self._return_guidance("none")

    def _is_occlusion_candidate(
        self,
        hand_center: tuple[int, int] | None,
        object_reference_center: tuple[int, int] | None,
    ) -> bool:
        if object_reference_center is None:
            return False

        effective_wait = self._effective_occlusion_wait_frames()
        candidate_hand = hand_center
        if (
            candidate_hand is None
            and self._last_hand_center is not None
            and self._lost_frames <= effective_wait + 2
        ):
            candidate_hand = self._last_hand_center
        if candidate_hand is None:
            return False

        distance = float(
            np.hypot(
                candidate_hand[0] - object_reference_center[0],
                candidate_hand[1] - object_reference_center[1],
            )
        )
        distance_threshold = float(self.occlusion_distance_threshold_px)
        if 0.0 < self._fps_ema <= 6.0:
            distance_threshold *= 1.35
        is_near_last_object = distance <= distance_threshold
        has_recent_overlap = (
            self._last_contact_ratio >= self.occlusion_overlap_threshold
            or self._overlap_evidence_frames > 0
        )
        has_recent_forward = (
            self._forward_evidence_frames > 0
            and self._lost_frames <= max(3, self.occlusion_stable_frames + 2)
        )
        if has_recent_overlap or has_recent_forward:
            return is_near_last_object
        return is_near_last_object and self._lost_frames >= effective_wait

    def _update_state_and_guidance(
        self,
        detected_object: DetectedObject | None,
        hand_center: tuple[int, int] | None,
        hand_bbox: tuple[int, int, int, int] | None,
        frame_shape: tuple[int, int],
    ) -> tuple[str, str]:
        frame_h, frame_w = frame_shape
        frame_center = (frame_w // 2, frame_h // 2)
        now = perf_counter()
        self._update_fps_estimate(now)

        if hand_center is not None:
            self._last_hand_center = hand_center

        if detected_object is None:
            self._lost_frames += 1
            if self._overlap_evidence_frames > 0:
                self._overlap_evidence_frames -= 1
            if self._forward_evidence_frames > 0:
                self._forward_evidence_frames -= 1
        else:
            self._lost_frames = 0
            self._last_object_center = detected_object.center
            self._last_visible_object_center = detected_object.center
            self._last_contact_ratio = estimate_contact_ratio(
                hand_bbox, detected_object.mask
            )
            if self._last_contact_ratio >= self.occlusion_overlap_threshold:
                self._overlap_evidence_frames = self._occlusion_overlap_memory_frames
            elif self._overlap_evidence_frames > 0:
                self._overlap_evidence_frames -= 1

        if (
            self._lost_frames >= self.track_lost_tolerance_frames
            and not self._occlusion_active
        ):
            self._set_state(ItemSearchState.SEGMENT)
            self._lost_frames = 0
            self._clear_occlusion()

        if not self.enable_item_search:
            return self._return_guidance("none")

        if self.state == ItemSearchState.SEGMENT:
            self._clear_occlusion()
            if detected_object is None:
                self._stable_detection_frames = 0
                return self._return_guidance("none")

            self._stable_detection_frames += 1
            if self._stable_detection_frames >= self.lock_required_frames:
                self._set_state(ItemSearchState.CENTER_GUIDE)
                return self._return_guidance("none")

            return self._return_guidance("none")

        if self.state == ItemSearchState.CENTER_GUIDE:
            self._clear_occlusion()
            if detected_object is None:
                return self._return_guidance("none")

            direction, centered, distance = get_center_guidance(
                detected_object.center,
                frame_center,
                self.center_threshold_px,
            )
            if centered:
                self._centered_frames += 1
                if self._centered_frames >= self.center_hold_frames:
                    self._set_state(ItemSearchState.TRACK)
                    return self._return_guidance("none")
                return self._return_guidance("none")

            self._centered_frames = 0
            return self._return_guidance(direction)

        if detected_object is None:
            reference_center = self._last_visible_object_center
            if reference_center is None:
                return self._return_guidance("none")

            candidate = self._is_occlusion_candidate(hand_center, reference_center)

            if candidate:
                self._occlusion_candidate_frames += 1
            else:
                self._occlusion_candidate_frames = 0

            required_stable_frames = self._effective_occlusion_stable_frames()
            if self._lost_frames >= self._effective_occlusion_wait_frames():
                # Đã chờ đủ frame reacquire mà vẫn mất vật -> ưu tiên tiến tay lên.
                required_stable_frames = 1
                if self._occlusion_candidate_frames == 0:
                    self._occlusion_candidate_frames = 1

            if (
                not self._occlusion_active
                and self._occlusion_candidate_frames >= required_stable_frames
            ):
                self._occlusion_active = True
                self._occlusion_active_since = now
                self._occlusion_reference_center = reference_center

            if self._occlusion_active:
                if self._occlusion_reference_center is None:
                    self._occlusion_reference_center = reference_center
                return self._return_guidance("forward")

            return self._return_guidance("none")

        contact_ratio = self._last_contact_ratio
        is_touching = contact_ratio >= self.contact_overlap_threshold

        if self._occlusion_active:
            reference_center = self._occlusion_reference_center
            self._clear_occlusion()
            if reference_center is not None:
                reappear_distance = float(
                    np.hypot(
                        detected_object.center[0] - reference_center[0],
                        detected_object.center[1] - reference_center[1],
                    )
                )
                if reappear_distance <= self.hand_guidance_threshold_px:
                    self._resume_forward_frames = self.forward_resume_hold_frames

        if self._resume_forward_frames > 0:
            self._resume_forward_frames -= 1
            return self._return_guidance("forward")

        if hand_center is None:
            return self._return_guidance("none")

        direction, _ = get_hand_guidance(
            hand_center,
            detected_object.center,
            threshold_px=self.hand_guidance_threshold_px,
            is_touching=is_touching,
        )

        if direction is None:
            return self._return_guidance("none")

        return self._return_guidance(direction)

    @staticmethod
    def _draw_object_overlay(
        frame: np.ndarray, detected_object: DetectedObject | None
    ) -> None:
        if detected_object is None:
            return
        overlay = frame.copy()
        overlay[detected_object.mask == 1] = (0, 255, 255)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0.0, frame)
        cv2.polylines(
            frame,
            [detected_object.contour.astype(np.int32)],
            True,
            (0, 255, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.circle(frame, detected_object.center, 5, (0, 255, 0), -1, cv2.LINE_AA)

    def process(self, frame: Any) -> Any:
        results = self.model(
            frame,
            verbose=False,
            device=self.device,
            half=self.use_fp16,
            imgsz=self.infer_imgsz,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            max_det=self.max_det,
        )
        annotated = np.ascontiguousarray(frame).copy()
        frame_h, frame_w = annotated.shape[:2]

        masks = self._extract_masks(results, frame_shape=(frame_h, frame_w))
        detected_object = self._select_target(masks)
        self._draw_object_overlay(annotated, detected_object)

        hand_points = []
        if self.hand_pose is not None:
            hand_points = self.hand_pose.detect(frame)
            annotated = self.hand_pose.draw(annotated, hand_points)
        primary_hand = extract_primary_hand(hand_points, frame_w, frame_h)

        guidance_text, command = self._update_state_and_guidance(
            detected_object=detected_object,
            hand_center=primary_hand.center,
            hand_bbox=primary_hand.bbox,
            frame_shape=(frame_h, frame_w),
        )

        if self.state == ItemSearchState.CENTER_GUIDE:
            draw_center_crosshair(annotated, (frame_w // 2, frame_h // 2))

        if primary_hand.center is not None and detected_object is not None:
            cv2.line(
                annotated,
                primary_hand.center,
                detected_object.center,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        prompt_text = ", ".join(self.yoloe_prompts) if self.yoloe_prompts else "none"
        cv2.putText(
            annotated,
            f"STATE: {self.state.value}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"PROMPT: {prompt_text}",
            (12, 54),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"GUIDE: {guidance_text}",
            (12, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"CMD: {command}",
            (12, 106),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        if self.hand_pose is None:
            cv2.putText(
                annotated,
                "MP OFF",
                (12, 132),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated

    def close(self) -> None:
        if self.hand_pose is not None:
            self.hand_pose.close()
