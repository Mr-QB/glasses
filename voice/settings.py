from dataclasses import dataclass


@dataclass(frozen=True)
class VoiceSettings:
    enabled: bool = False
    language: str = "vi-VN"
    rate: int = 165
    volume: float = 1.0
    fallback_to_console: bool = True


@dataclass(frozen=True)
class STTSettings:
    model_id: str = "vinai/PhoWhisper-large"
    device_preference: str = "auto"
    language: str = "vi"
    task: str = "transcribe"
    target_sample_rate: int = 16000
    chunk_length_s: int = 30
    return_timestamps: bool = False
    use_flash_attention_2: bool = False
