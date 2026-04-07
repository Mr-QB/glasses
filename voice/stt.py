from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any
import warnings

import numpy as np

from voice.settings import STTSettings


@dataclass(frozen=True)
class STTResult:
    text: str
    language: str
    source: str


class PhoWhisperSTT:
    def __init__(self, settings: STTSettings | None = None) -> None:
        self.settings = settings or STTSettings()
        self._pipe = None
        self._load_error: str | None = None

    @property
    def ready(self) -> bool:
        return self._pipe is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        if self._pipe is not None:
            return

        try:
            torch = importlib.import_module("torch")
            transformers = importlib.import_module("transformers")
            auto_model = getattr(transformers, "AutoModelForSpeechSeq2Seq")
            auto_processor = getattr(transformers, "AutoProcessor")
            asr_pipeline = getattr(transformers, "pipeline")

            pref = (self.settings.device_preference or "auto").strip().lower()
            if pref == "cpu":
                use_cuda = False
            elif pref in {"cuda", "gpu"}:
                use_cuda = bool(torch.cuda.is_available())
            else:
                use_cuda = torch.cuda.is_available()
            dtype = torch.float16 if use_cuda else torch.float32
            device = "cuda:0" if use_cuda else "cpu"

            model_kwargs: dict[str, Any] = {
                "dtype": dtype,
                "low_cpu_mem_usage": True,
                # Some HF repos disable discussions; forcing safetensors can trigger
                # auto-conversion logic that queries discussions and fails with 403.
                "use_safetensors": False,
            }
            if self.settings.use_flash_attention_2:
                model_kwargs["attn_implementation"] = "flash_attention_2"

            # Prefer local cache to avoid extra hub calls and noisy background logs.
            model = self._load_with_local_first(
                auto_model,
                self.settings.model_id,
                **model_kwargs,
            )
            processor = self._load_with_local_first(
                auto_processor,
                self.settings.model_id,
            )

            # Keep runtime logs concise while preserving actual failures.
            warnings.filterwarnings(
                "ignore",
                message="Using `chunk_length_s` is very experimental",
            )

            self._pipe = asr_pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=processor.tokenizer,
                feature_extractor=processor.feature_extractor,
                dtype=dtype,
                device=device,
                chunk_length_s=self.settings.chunk_length_s,
                ignore_warning=True,
            )
            self._load_error = None
        except Exception as exc:
            self._load_error = str(exc)
            raise RuntimeError(f"Failed to initialize PhoWhisper STT: {exc}") from exc

    def transcribe_file(self, audio_path: str | Path) -> STTResult:
        file_path = Path(audio_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        self.load()
        assert self._pipe is not None

        result = self._run_inference(str(file_path))
        return STTResult(
            text=self._extract_text(result),
            language=self.settings.language,
            source=str(file_path),
        )

    def transcribe_array(self, audio: np.ndarray, sample_rate: int) -> STTResult:
        if audio.ndim != 1:
            raise ValueError("Expected mono audio array with shape (n_samples,)")

        self.load()
        assert self._pipe is not None

        payload = {
            "array": audio.astype(np.float32),
            "sampling_rate": int(sample_rate),
        }
        result = self._run_inference(payload)
        return STTResult(
            text=self._extract_text(result),
            language=self.settings.language,
            source="audio-array",
        )

    def transcribe_microphone(self, seconds: float = 5.0) -> STTResult:
        try:
            import sounddevice as sd
        except Exception as exc:
            raise RuntimeError(
                "sounddevice is required for microphone capture. Install voice/requirements.txt"
            ) from exc

        sr = self.settings.target_sample_rate
        frames = max(1, int(seconds * sr))
        recording = sd.rec(frames, samplerate=sr, channels=1, dtype="float32")
        sd.wait()
        mono = np.squeeze(recording, axis=1)
        return self.transcribe_array(mono, sr)

    @staticmethod
    def _extract_text(result: Any) -> str:
        if isinstance(result, dict) and "text" in result:
            return str(result["text"]).strip()
        return str(result).strip()

    @staticmethod
    def _load_with_local_first(auto_class, model_id: str, **kwargs):
        try:
            return auto_class.from_pretrained(model_id, local_files_only=True, **kwargs)
        except Exception:
            return auto_class.from_pretrained(model_id, **kwargs)

    def _run_inference(self, audio_input: Any) -> Any:
        assert self._pipe is not None

        kwargs: dict[str, Any] = {
            "generate_kwargs": {
                "language": self.settings.language,
                "task": self.settings.task,
            }
        }
        if self.settings.return_timestamps:
            kwargs["return_timestamps"] = True

        return self._pipe(audio_input, **kwargs)
