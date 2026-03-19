from dataclasses import dataclass


@dataclass(frozen=True)
class VisionSettings:
    stream_url: str = "http://100.123.237.23:8080/video"
    model_path: str = r"models\yolo26m.pt"
    window_name: str = "AI Glasses Camera"
    frame_width: int = 640
    frame_height: int = 480
    queue_size: int = 2
    yolo_device: str = "auto"
    use_fp16: bool = True
    enable_hand_pose: bool = True
    max_num_hands: int = 2
    hand_detection_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
