"""
AI Glasses Voice Server Node - STT + Ollama Processing

Runs on remote server:
    - Receives WAV audio from laptop via /transcribe endpoint
    - Performs STT (Speech-to-Text) using PhoWhisper
    - Extracts object labels using Ollama
    - Returns extraction result directly in /transcribe response

Usage:
    python main_voice_server.py \
        --server-host 0.0.0.0 \
        --server-port 5051

Endpoints:
    POST /transcribe - Receive WAV, perform STT + Ollama, return response only
    GET  /health     - Health check
"""

import argparse
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

from voice.assistant import VoiceAssistant
from voice.settings import STTSettings
from voice.ollama_object_extractor import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    call_ollama,
)
from shared.target_handoff import coerce_target_handoff


class VoiceServerNode:
    """Remote voice server node that processes audio and sends labels back to laptop."""

    def __init__(
        self,
        server_host: str = "0.0.0.0",
        server_port: int = 5051,
        save_dir: str = "raw_data/http_uploads",
        ollama_model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_timeout_seconds: float = 20.0,
        ollama_max_output_tokens: int = 96,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        base_dir = Path(__file__).resolve().parent
        configured_save_dir = Path(save_dir)
        self.save_dir = (
            configured_save_dir
            if configured_save_dir.is_absolute()
            else (base_dir / configured_save_dir)
        )
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.assistant = VoiceAssistant(
            stt_settings=STTSettings(device_preference="cuda"),
        )
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.ollama_timeout_seconds = max(1.0, float(ollama_timeout_seconds))
        self.ollama_max_output_tokens = max(16, int(ollama_max_output_tokens))

        self.app = self._create_app()

    def _create_app(self) -> Flask:
        app = Flask(__name__)
        self.register_routes(app)
        return app

    def register_routes(self, app: Flask) -> None:
        @app.get("/health")
        def health() -> tuple[dict, int]:
            return jsonify({"status": "ok", "node": "voice_server"}), 200

        @app.post("/transcribe")
        def transcribe() -> tuple[dict, int]:
            """Receive WAV audio, perform STT + Ollama, return response only."""
            request_id = request.headers.get("X-Request-ID", "unknown")
            try:
                audio_path = self._extract_audio_from_request(request_id)
            except ValueError as exc:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "code": "INVALID_AUDIO_REQUEST",
                            "error": str(exc),
                            "request_id": request_id,
                        }
                    ),
                    400,
                )
            except OSError as exc:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "code": "AUDIO_SAVE_FAILED",
                            "error": str(exc),
                            "request_id": request_id,
                            "save_dir": str(self.save_dir),
                        }
                    ),
                    500,
                )

            try:
                result_payload, has_label = self._process_audio_path(
                    audio_path, request_id
                )
                return jsonify(result_payload), (200 if has_label else 422)
            except Exception as exc:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "code": "VOICE_PROCESSING_FAILED",
                            "error": str(exc),
                            "request_id": request_id,
                        }
                    ),
                    500,
                )

    def _process_audio_path(
        self, audio_path: Path, request_id: str
    ) -> tuple[dict, bool]:
        """Process WAV file: STT + Ollama, return payload."""
        print(f"[VOICE-STT@{request_id}] Processing {audio_path}")

        result = self.assistant.transcribe_file(audio_path)
        print(f"[VOICE-STT@{request_id}] Transcript: {result.text}")

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

        if target.normalized_label:
            print(
                f"[VOICE->LABEL@{request_id}] label={target.normalized_label} "
                f"confidence={target.confidence:.2f}"
            )
        else:
            print(f"[VOICE->LABEL@{request_id}] no label (reason: {target.reason})")

        has_label = bool(target.normalized_label)
        response_payload = {
            "status": "ok" if has_label else "no_label",
            "code": "LABEL_EXTRACTED" if has_label else "NO_LABEL_EXTRACTED",
            "request_id": request_id,
            "text": result.text,
            "language": result.language,
            "source": result.source,
            "message": (
                "Label extracted successfully"
                if has_label
                else "STT completed but no reliable object label was extracted"
            ),
            "target": {
                "label": target.label,
                "normalized_label": target.normalized_label,
                "confidence": target.confidence,
                "reason": target.reason,
                "extraction_error": extraction_error,
            },
        }

        if not has_label:
            print(f"[VOICE@{request_id}] No label extracted; response-only mode")

        return response_payload, has_label

    def _extract_audio_from_request(self, request_id: str) -> Path:
        data = request.get_data(cache=False)
        if not data:
            raise ValueError("Audio body is empty")

        file_name = f"audio_{request_id}.wav"
        return self._write_audio(file_name, data)

    def _write_audio(self, file_name: str, audio_bytes: bytes) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = self.save_dir / f"{stamp}_{file_name}"
        out_path.write_bytes(audio_bytes)
        return out_path

    def serve_forever(self) -> None:
        print(f"[VOICE-HTTP] Listening on http://{self.server_host}:{self.server_port}")
        self.assistant.preload_stt()
        self.app.run(
            host=self.server_host,
            port=self.server_port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )


def main():
    parser = argparse.ArgumentParser(
        description="AI Glasses Voice Server Node (STT + Ollama)"
    )
    parser.add_argument("--server-host", default="0.0.0.0", help="Voice server host")
    parser.add_argument(
        "--server-port", type=int, default=5051, help="Voice server port"
    )
    parser.add_argument(
        "--save-dir",
        default="raw_data/http_uploads",
        help="Directory to save uploaded audio files",
    )
    args = parser.parse_args()

    try:
        node = VoiceServerNode(
            server_host=args.server_host,
            server_port=args.server_port,
            save_dir=args.save_dir,
        )
    except RuntimeError as exc:
        print(f"Failed to initialize voice server: {exc}")
        return

    print("=" * 70)
    print("🎤 VOICE SERVER NODE STARTED")
    print("=" * 70)
    print(f"Voice service:      http://{args.server_host}:{args.server_port}")
    print(
        f"Transcribe:         POST http://{args.server_host}:{args.server_port}/transcribe"
    )
    print("Mode:               response-only (no callback)")
    print(f"Health check:       http://{args.server_host}:{args.server_port}/health")
    print("Press Ctrl + C để thoát.")
    print("=" * 70)

    try:
        node.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
