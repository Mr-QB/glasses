from __future__ import annotations

from typing import List, Tuple
import os
import warnings

import cv2


class HandPoseProcessor:
    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.landmarker = None
        self.status_message = "Hand pose not initialized."
        self.connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
        ]

        try:
            import mediapipe as mp
        except ImportError:
            self.status_message = "MediaPipe not installed. Hand tracking is disabled. Install with: pip install mediapipe"
            warnings.warn(self.status_message)
            return

        tasks = getattr(mp, "tasks", None)
        if tasks is None:
            self.status_message = (
                "MediaPipe tasks API not available. Hand tracking is disabled."
            )
            warnings.warn(self.status_message)
            return

        model_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "hand_landmarker.task"
        )
        model_path = os.path.abspath(model_path)

        if not os.path.exists(model_path):
            self.status_message = f"Hand landmarker model not found at {model_path}. Hand tracking is disabled."
            warnings.warn(self.status_message)
            return

        base_options = tasks.BaseOptions(model_asset_path=model_path)

        options = tasks.vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = tasks.vision.HandLandmarker.create_from_options(options)
        self._mp = mp
        self.status_message = "MediaPipe hand tracking is enabled."

    def is_enabled(self) -> bool:
        return self.landmarker is not None

    def get_status(self) -> str:
        return self.status_message

    def detect(self, frame) -> List[List[Tuple[float, float]]]:
        if self.landmarker is None:
            return []

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
        )
        results = self.landmarker.detect(mp_image)

        if not results.hand_landmarks:
            return []

        hand_points: List[List[Tuple[float, float]]] = []
        for hand_landmarks in results.hand_landmarks:
            points = [(lm.x, lm.y) for lm in hand_landmarks]
            hand_points.append(points)

        return hand_points

    def draw(self, frame, hand_points: List[List[Tuple[float, float]]]):
        height, width = frame.shape[:2]

        for points in hand_points:
            for start_idx, end_idx in self.connections:
                x1 = int(points[start_idx][0] * width)
                y1 = int(points[start_idx][1] * height)
                x2 = int(points[end_idx][0] * width)
                y2 = int(points[end_idx][1] * height)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

            for x_norm, y_norm in points:
                x = int(x_norm * width)
                y = int(y_norm * height)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

        return frame

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()
