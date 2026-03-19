import cv2
from queue import Empty, Full, Queue
from threading import Thread
from time import perf_counter
from typing import List, Optional, Tuple

from vision.camera import open_camera
from vision.hand_pose import HandPoseProcessor
from vision.processor import FrameProcessor
from vision.settings import VisionSettings


class VisionPipeline:
    def __init__(self, settings: VisionSettings) -> None:
        self.settings = settings
        self.cap = open_camera(
            settings.stream_url,
            frame_width=settings.frame_width,
            frame_height=settings.frame_height,
        )
        self.processor = FrameProcessor(
            model_path=settings.model_path,
            device=settings.yolo_device,
            use_fp16=settings.use_fp16,
        )
        self.hand_processor = HandPoseProcessor(
            max_num_hands=settings.max_num_hands,
            min_detection_confidence=settings.hand_detection_confidence,
            min_tracking_confidence=settings.hand_tracking_confidence,
        )
        self.raw_frames_yolo: Queue = Queue(maxsize=settings.queue_size)
        self.raw_frames_hands: Queue = Queue(maxsize=settings.queue_size)
        self.processed_frames: Queue = Queue(maxsize=settings.queue_size)
        self.hand_results: Queue = Queue(maxsize=settings.queue_size)
        self._running = False
        self._read_thread: Optional[Thread] = None
        self._yolo_thread: Optional[Thread] = None
        self._hand_thread: Optional[Thread] = None
        self._latest_hands: List[List[Tuple[float, float]]] = []

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._read_thread = Thread(target=self._read_loop, daemon=True)
        self._yolo_thread = Thread(target=self._yolo_loop, daemon=True)
        self._hand_thread = Thread(target=self._hand_loop, daemon=True)
        self._read_thread.start()
        self._yolo_thread.start()
        self._hand_thread.start()

    @staticmethod
    def _push_latest(queue: Queue, item) -> None:
        if queue.full():
            try:
                queue.get_nowait()
            except Empty:
                pass

        try:
            queue.put_nowait(item)
        except Full:
            pass

    def _read_loop(self) -> None:
        while self._running:
            ret, frame = self.cap.read()
            if not ret:
                self._running = False
                break

            frame = cv2.resize(
                frame,
                (self.settings.frame_width, self.settings.frame_height),
                interpolation=cv2.INTER_AREA,
            )

            self._push_latest(self.raw_frames_yolo, frame.copy())
            self._push_latest(self.raw_frames_hands, frame)

    def _yolo_loop(self) -> None:
        last_log_time = perf_counter()
        frame_count = 0

        while self._running or not self.raw_frames_yolo.empty():
            try:
                frame = self.raw_frames_yolo.get(timeout=0.2)
            except Empty:
                continue

            annotated = self.processor.process(frame)
            self._push_latest(self.processed_frames, annotated)

            frame_count += 1
            now = perf_counter()
            elapsed = now - last_log_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(f"Current FPS: {fps:.2f}")
                frame_count = 0
                last_log_time = now

    def _hand_loop(self) -> None:
        while self._running or not self.raw_frames_hands.empty():
            try:
                frame = self.raw_frames_hands.get(timeout=0.2)
            except Empty:
                continue

            hand_points = self.hand_processor.detect(frame)
            self._push_latest(self.hand_results, hand_points)

    def next_frame(self):
        while self._running or not self.processed_frames.empty():
            try:
                frame = self.processed_frames.get(timeout=0.2)
            except Empty:
                continue

            while True:
                try:
                    self._latest_hands = self.hand_results.get_nowait()
                except Empty:
                    break

            if self.settings.enable_hand_pose and self._latest_hands:
                frame = self.hand_processor.draw(frame, self._latest_hands)

            return frame

        return None

    def stop(self) -> None:
        self._running = False

        if self._read_thread is not None:
            self._read_thread.join(timeout=1.0)
        if self._yolo_thread is not None:
            self._yolo_thread.join(timeout=1.0)
        if self._hand_thread is not None:
            self._hand_thread.join(timeout=1.0)

        self.cap.release()
        self.hand_processor.close()
        cv2.destroyAllWindows()
