from __future__ import annotations

from pathlib import Path

from voice.settings import STTSettings
from voice.settings import VoiceSettings
from voice.stt import PhoWhisperSTT
from voice.stt import STTResult
from voice.tts import SpeechSynthesizer


class VoiceAssistant:
    def __init__(
        self,
        settings: VoiceSettings | None = None,
        stt_settings: STTSettings | None = None,
    ) -> None:
        self.settings = settings or VoiceSettings(enabled=True)
        self.stt_settings = stt_settings or STTSettings()
        self.synth: SpeechSynthesizer | None = None
        if self.settings.enabled:
            self.synth = SpeechSynthesizer(
                rate=self.settings.rate,
                volume=self.settings.volume,
                fallback_to_console=self.settings.fallback_to_console,
            )
        self.stt = PhoWhisperSTT(self.stt_settings)
        self._started = False

    def start(self) -> None:
        if not self.settings.enabled or self.synth is None:
            return

        self.synth.start()
        self._started = True

        if self.synth.available:
            self.speak("Trợ lý giọng nói đã sẵn sàng")
        else:
            reason = self.synth.engine_error or "không rõ lý do"
            print(f"[VOICE] TTS chưa khả dụng: {reason}")
            if self.settings.fallback_to_console:
                self.speak("Đã bật chế độ thông báo bằng console")

    def stop(self) -> None:
        if not self._started or self.synth is None:
            return

        self.speak("Đang tắt trợ lý giọng nói")
        self.synth.stop()
        self._started = False

    def speak(self, message: str) -> None:
        if not self.settings.enabled or self.synth is None:
            return
        self.synth.speak(message)

    def transcribe_file(self, audio_path: str | Path) -> STTResult:
        return self.stt.transcribe_file(audio_path)

    def transcribe_microphone(self, seconds: float = 5.0) -> STTResult:
        return self.stt.transcribe_microphone(seconds=seconds)

    def preload_stt(self) -> None:
        self.stt.load()
