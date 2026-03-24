import cv2
import traceback
from queue import Empty, Full, Queue
from threading import Thread
from time import perf_counter
from typing import Optional

from vision.camera import open_camera
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
            infer_imgsz=settings.infer_imgsz,
            conf_threshold=settings.conf_threshold,
            iou_threshold=settings.iou_threshold,
            max_det=settings.max_det,
            yoloe_prompts=settings.yoloe_prompts,
            enable_hand_pose=settings.enable_hand_pose,
            max_num_hands=settings.max_num_hands,
            hand_detection_confidence=settings.hand_detection_confidence,
            hand_tracking_confidence=settings.hand_tracking_confidence,
            enable_item_search=settings.enable_item_search,
            lock_required_frames=settings.lock_required_frames,
            center_threshold_px=settings.center_threshold_px,
            center_hold_frames=settings.center_hold_frames,
            hand_guidance_threshold_px=settings.hand_guidance_threshold_px,
            track_lost_tolerance_frames=settings.track_lost_tolerance_frames,
            contact_overlap_threshold=settings.contact_overlap_threshold,
        )
        self.raw_frames_segmentation: Queue = Queue(maxsize=settings.queue_size)
        self.processed_frames: Queue = Queue(maxsize=settings.queue_size)
        self._running = False
        self._read_thread: Optional[Thread] = None
        self._segmentation_thread: Optional[Thread] = None
        self._worker_error: Optional[str] = None
        self._error_reported = False

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._read_thread = Thread(target=self._read_loop, daemon=True)
        self._segmentation_thread = Thread(target=self._segmentation_loop, daemon=True)
        self._read_thread.start()
        self._segmentation_thread.start()

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

            target_size = (self.settings.frame_width, self.settings.frame_height)
            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(
                    frame,
                    target_size,
                    interpolation=cv2.INTER_AREA,
                )

            self._push_latest(self.raw_frames_segmentation, frame)

    def _segmentation_loop(self) -> None:
        last_log_time = perf_counter()
        frame_count = 0

        while self._running or not self.raw_frames_segmentation.empty():
            try:
                frame = self.raw_frames_segmentation.get(timeout=0.2)
            except Empty:
                continue

            try:
                start_time = perf_counter()
                annotated = self.processor.process(frame)
                process_time = perf_counter() - start_time
            except Exception as exc:
                self._worker_error = f"[YOLOE] Segmentation worker crashed: {exc}"
                print(self._worker_error)
                print(traceback.format_exc())
                self._running = False
                break

            self._push_latest(self.processed_frames, annotated)

            frame_count += 1
            now = perf_counter()
            elapsed = now - last_log_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                print(
                    f"[YOLOE] Segmentation FPS: {fps:.2f} | Avg latency: {(elapsed/frame_count)*1000:.1f}ms | Processed frames queue: {self.processed_frames.qsize()}"
                )
                frame_count = 0
                last_log_time = now

    def next_frame(self):
        while self._running or not self.processed_frames.empty():
            try:
                frame = self.processed_frames.get(timeout=0.2)
            except Empty:
                if self._worker_error and not self._error_reported:
                    print(self._worker_error)
                    self._error_reported = True
                continue

            return frame

        if self._worker_error and not self._error_reported:
            print(self._worker_error)
            self._error_reported = True

        return None

    def get_last_error(self) -> Optional[str]:
        return self._worker_error

    def stop(self) -> None:
        self._running = False

        if self._read_thread is not None:
            self._read_thread.join(timeout=1.0)
        if self._segmentation_thread is not None:
            self._segmentation_thread.join(timeout=1.0)

        self.processor.close()
        self.cap.release()
        cv2.destroyAllWindows()
