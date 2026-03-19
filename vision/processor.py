from typing import Any

import torch
from ultralytics import YOLO


class FrameProcessor:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        device: str = "auto",
        use_fp16: bool = True,
    ) -> None:
        self.model = YOLO(model_path)
        self.device = self._resolve_device(device)
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")

        self.model.to(self.device)
        print(f"YOLO device: {self.device} | fp16: {self.use_fp16}")

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def process(self, frame: Any) -> Any:
        results = self.model(
            frame,
            verbose=False,
            device=self.device,
            half=self.use_fp16,
        )
        return results[0].plot()
