from dataclasses import dataclass


@dataclass(frozen=True)
class VisionSettings:
    stream_url: str = "http://172.20.10.2/stream"
    model_path: str = "models/yoloe-11l-seg.pt"
    window_name: str = "AI Glasses Camera"
    frame_width: int = 640
    frame_height: int = 480
    queue_size: int = 2
    yolo_device: str = "auto"
    use_fp16: bool = True
    infer_imgsz: int = 416
    conf_threshold: float = 0.4
    iou_threshold: float = 0.6
    max_det: int = 50
    yoloe_prompts: tuple[str, ...] = ("cell phone",)
    enable_hand_pose: bool = True
    max_num_hands: int = 2
    hand_detection_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
    enable_item_search: bool = True
    lock_required_frames: int = 8
    center_threshold_px: int = 35
    center_hold_frames: int = 6
    hand_guidance_threshold_px: int = 35
    track_lost_tolerance_frames: int = 10
    contact_overlap_threshold: float = 0.12
    occlusion_distance_threshold_px: int = 80
    occlusion_overlap_threshold: float = 0.12
    occlusion_stable_frames: int = 2
    occlusion_reacquire_wait_frames: int = 3
    occlusion_timeout_seconds: float = 3.0
    forward_resume_hold_frames: int = 2
    active_keepalive_seconds: float = 45.0
    read_fail_reconnect_threshold: int = 8
    reconnect_initial_delay_seconds: float = 0.5
    reconnect_max_delay_seconds: float = 5.0
