from dataclasses import dataclass


@dataclass(frozen=True)
class VisionSettings:
    stream_url: str = "http://100.123.237.23:8080/video"
    model_path: str = "model/yoloe-11l-seg.pt"
    window_name: str = "AI Glasses Camera"
    frame_width: int = 640
    frame_height: int = 480
    queue_size: int = 3
    yolo_device: str = "auto"
    use_fp16: bool = True
    infer_imgsz: int = 416
    conf_threshold: float = 0.4
    iou_threshold: float = 0.6
    max_det: int = 50
    yoloe_prompts: tuple[str, ...] = ("bottled water",)
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
