import cv2
import traceback
from queue import Empty, Full, Queue
from threading import RLock
from threading import Thread
from time import perf_counter
from time import sleep
from typing import Callable, Optional

from vision.camera import open_camera
from vision.processor import FrameProcessor
from vision.settings import VisionSettings
from shared.target_handoff import TargetHandoffBus


class VisionPipeline:
    def __init__(
        self,
        settings: VisionSettings,
        on_guidance_change: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.target_bus: TargetHandoffBus | None = None
        self._active = False
        self._active_keepalive_s = max(0.0, float(settings.active_keepalive_seconds))
        self._last_activity_ts = 0.0
        self._activity_lock = RLock()
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
            occlusion_distance_threshold_px=settings.occlusion_distance_threshold_px,
            occlusion_overlap_threshold=settings.occlusion_overlap_threshold,
            occlusion_stable_frames=settings.occlusion_stable_frames,
            occlusion_reacquire_wait_frames=settings.occlusion_reacquire_wait_frames,
            occlusion_timeout_seconds=settings.occlusion_timeout_seconds,
            forward_resume_hold_frames=settings.forward_resume_hold_frames,
            on_guidance_change=on_guidance_change,
        )
        self.raw_frames_segmentation: Queue = Queue(maxsize=settings.queue_size)
        self.processed_frames: Queue = Queue(maxsize=settings.queue_size)
        self._running = False
        self._read_thread: Optional[Thread] = None
        self._segmentation_thread: Optional[Thread] = None
        self._worker_error: Optional[str] = None
        self._error_reported = False
        self._consecutive_read_failures = 0

    def attach_target_bus(self, target_bus: TargetHandoffBus | None) -> None:
        self.target_bus = target_bus

    def apply_target_handoff(self, target) -> None:
        if target is None or not getattr(target, "normalized_label", None):
            return

        normalized_label = str(target.normalized_label).strip()
        if not normalized_label:
            return

        print(
            f"[YOLOE] apply target immediately: "
            f"label={getattr(target, 'label', None)!r} "
            f"normalized_label={normalized_label!r} "
            f"confidence={getattr(target, 'confidence', 0.0):.2f}"
        )

        if self.target_bus is not None:
            self.target_bus.publish(target)

        self.processor.set_prompts([normalized_label])
        self.activate(self.settings.active_keepalive_seconds)

    def activate(self, keepalive_seconds: float | None = None) -> None:
        with self._activity_lock:
            if keepalive_seconds is not None:
                self._active_keepalive_s = max(0.0, float(keepalive_seconds))
            self._active = True
            self._last_activity_ts = perf_counter()

    def deactivate(self) -> None:
        with self._activity_lock:
            self._active = False
            self._last_activity_ts = 0.0
        self._clear_queue(self.raw_frames_segmentation)
        self._clear_queue(self.processed_frames)

    def is_active(self) -> bool:
        with self._activity_lock:
            return self._active

    def _refresh_active_state(self) -> bool:
        with self._activity_lock:
            if not self._active:
                return False

            keepalive = self._active_keepalive_s
            if keepalive <= 0:
                return True

            if perf_counter() - self._last_activity_ts <= keepalive:
                return True

            self._active = False
            self._last_activity_ts = 0.0

        self._clear_queue(self.raw_frames_segmentation)
        self._clear_queue(self.processed_frames)
        return False

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

    @staticmethod
    def _clear_queue(queue: Queue) -> None:
        while True:
            try:
                queue.get_nowait()
            except Empty:
                return

    def _reopen_camera_with_backoff(self) -> bool:
        delay_s = max(0.1, float(self.settings.reconnect_initial_delay_seconds))
        max_delay_s = max(delay_s, float(self.settings.reconnect_max_delay_seconds))

        while self._running:
            try:
                self.cap.release()
            except Exception:
                pass

            try:
                self.cap = open_camera(
                    self.settings.stream_url,
                    frame_width=self.settings.frame_width,
                    frame_height=self.settings.frame_height,
                )
                self._consecutive_read_failures = 0
                print("[CAMERA] Reconnected stream successfully.")
                return True
            except Exception as exc:
                print(
                    f"[CAMERA] Reconnect failed: {exc}. " f"Retrying in {delay_s:.1f}s"
                )
                sleep(delay_s)
                delay_s = min(max_delay_s, delay_s * 2.0)

        return False

    def _read_loop(self) -> None:
        while self._running:
            if not self._refresh_active_state():
                sleep(0.05)
                continue

            ret, frame = self.cap.read()
            if not ret:
                self._consecutive_read_failures += 1
                if (
                    self._consecutive_read_failures
                    >= self.settings.read_fail_reconnect_threshold
                ):
                    print(
                        "[CAMERA] Read failures reached threshold "
                        f"({self._consecutive_read_failures}). Attempting reconnect..."
                    )
                    if not self._reopen_camera_with_backoff():
                        break
                    continue

                sleep(0.02)
                continue

            self._consecutive_read_failures = 0

            # Preprocess: rotate frame 90 degrees clockwise before downstream processing.
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

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
            if not self._refresh_active_state():
                sleep(0.05)
                continue

            try:
                frame = self.raw_frames_segmentation.get(timeout=0.2)
            except Empty:
                continue

            if self.target_bus is not None:
                target = self.target_bus.consume_latest()
                if target is not None and target.normalized_label:
                    print(
                        "[YOLOE] apply extracted label="
                        f"{target.normalized_label}"
                        f" | confidence={target.confidence:.2f}"
                    )
                    self.processor.set_prompts([target.normalized_label])

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
