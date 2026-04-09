from __future__ import annotations

import base64
from datetime import datetime
from pathlib import Path
from typing import Callable

from flask import Flask
from flask import jsonify
from flask import request

from shared.target_handoff import TargetHandoffBus
from shared.target_handoff import coerce_target_handoff
from voice.assistant import VoiceAssistant
from voice.ollama_object_extractor import DEFAULT_MODEL
from voice.ollama_object_extractor import DEFAULT_OLLAMA_URL
from voice.ollama_object_extractor import call_ollama


class VoiceHTTPServer:
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5052,
        save_dir: str = "raw_data/http_uploads",
        assistant: VoiceAssistant | None = None,
        target_bus: TargetHandoffBus | None = None,
        ollama_model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_timeout_seconds: float = 20.0,
        ollama_max_output_tokens: int = 96,
        on_audio_received: Callable[[], None] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.assistant = assistant or VoiceAssistant()
        self.target_bus = target_bus
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.ollama_timeout_seconds = max(1.0, float(ollama_timeout_seconds))
        self.ollama_max_output_tokens = max(16, int(ollama_max_output_tokens))
        self.on_audio_received = on_audio_received

        self.app = self._create_app()

    def serve_forever(self) -> None:
        print(f"[VOICE-HTTP] Listening on http://{self.host}:{self.port}")
        self.assistant.preload_stt()
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )

    def _create_app(self) -> Flask:
        app = Flask(__name__)
        self.register_routes(app)
        return app

    def register_routes(self, app: Flask) -> None:
        @app.get("/health")
        def health() -> tuple[str, int]:
            return jsonify({"status": "ok"}), 200

        @app.post("/transcribe")
        def transcribe() -> tuple[str, int]:
            try:
                audio_path = self._extract_audio_from_request()
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400

            if self.on_audio_received is not None:
                self.on_audio_received()

            try:
                payload = self._process_audio_upload(audio_path)
            except ValueError as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception as exc:
                return jsonify({"error": str(exc)}), 500

            if self.on_audio_received is not None:
                self.on_audio_received()

            return jsonify(payload), 200

    def _process_audio_upload(self, audio_path: Path) -> dict:
        if audio_path.suffix.lower() != ".wav":
            raise ValueError("Only WAV uploads are supported for /transcribe")

        result = self.assistant.transcribe_file(audio_path)

        print(f"[VOICE-STT] transcript: {result.text}")

        extraction: dict[str, object]
        extraction_error: str | None = None

        transcript_text = result.text.strip()
        if len(transcript_text) < 4:
            extraction = {
                "object": None,
                "normalized_object": None,
                "confidence": 0.0,
                "reason": "Transcript too short for reliable extraction",
            }
        else:
            try:
                extraction = call_ollama(
                    transcript_text,
                    self.ollama_model,
                    self.ollama_url,
                    timeout_seconds=self.ollama_timeout_seconds,
                    max_output_tokens=self.ollama_max_output_tokens,
                )
            except Exception as exc:
                extraction = {
                    "object": None,
                    "normalized_object": None,
                    "confidence": 0.0,
                    "reason": f"Ollama extraction failed: {exc}",
                }
                extraction_error = str(exc)

        target = coerce_target_handoff(
            transcript_text,
            extraction,
            source_audio=result.source,
        )

        published = False
        if self.target_bus is not None and target.normalized_label:
            self.target_bus.publish(target)
            published = True

        if target.normalized_label:
            print(
                "[VOICE->YOLOE] extracted label="
                f"{target.normalized_label}"
                f" | confidence={target.confidence:.2f}"
                f" | published={published}"
            )
        else:
            print(
                "[VOICE->YOLOE] no label extracted"
                f" | reason={target.reason or 'unknown'}"
            )

        return {
            "text": result.text,
            "language": result.language,
            "source": result.source,
            "target": {
                "label": target.label,
                "normalized_label": target.normalized_label,
                "confidence": target.confidence,
                "reason": target.reason,
                "published": published,
                "extraction_error": extraction_error,
            },
        }

    def _extract_audio_from_request(self) -> Path:
        if "audio" in request.files:
            file = request.files["audio"]
            if not file.filename:
                raise ValueError("Uploaded file must have a filename")
            file_name = self._safe_filename(file.filename)
            data = file.read()
            if not data:
                raise ValueError("Uploaded file is empty")
            return self._write_audio(file_name, data)

        if request.is_json:
            payload = request.get_json(silent=True) or {}
            audio_b64 = str(payload.get("audio_base64", ""))
            if not audio_b64:
                raise ValueError("audio_base64 is required in JSON payload")
            try:
                data = base64.b64decode(audio_b64)
            except Exception as exc:
                raise ValueError("audio_base64 is invalid") from exc
            file_name = self._safe_filename(str(payload.get("filename", "audio.wav")))
            return self._write_audio(file_name, data)

        data = request.get_data(cache=False)
        if not data:
            raise ValueError("Request body is empty")

        file_name = self._infer_raw_upload_filename()
        return self._write_audio(file_name, data)

    def _infer_raw_upload_filename(self) -> str:
        header_name = request.headers.get("X-Filename", "")
        if header_name:
            safe_name = self._safe_filename(header_name)
            if Path(safe_name).suffix.lower() == ".wav":
                return safe_name
            return f"{Path(safe_name).stem}.wav"

        return "audio.wav"

    @staticmethod
    def _safe_filename(name: str) -> str:
        return Path(str(name)).name or "audio.wav"

    def _write_audio(self, file_name: str, audio_bytes: bytes) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = self.save_dir / f"{stamp}_{file_name}"
        out_path.write_bytes(audio_bytes)
        return out_path
