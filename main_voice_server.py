"""
AI Glasses Voice Server Node - STT + Ollama Processing

Runs on remote server:
  - Receives WAV audio from laptop via /transcribe endpoint
  - Performs STT (Speech-to-Text) using PhoWhisper
  - Extracts object labels using Ollama
  - Sends label callbacks back to laptop /target_callback

Usage:
  python main_voice_server.py \\n    --server-host 0.0.0.0 \\n    --server-port 5051 \\n    --laptop-callback-url http://127.0.0.1:5051/target_callback

Endpoints:
  POST /transcribe - Receive WAV, perform STT + Ollama, push callback
  GET  /health     - Health check
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
        laptop_callback_url: str = "",
        ollama_model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        ollama_timeout_seconds: float = 20.0,
        ollama_max_output_tokens: int = 96,
    ) -> None:
        self.server_host = server_host
        self.server_port = server_port
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.laptop_callback_url = laptop_callback_url.rstrip("/")
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
            """Receive WAV audio, perform STT + Ollama, send callback to laptop."""
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
        """Process WAV file: STT + Ollama, then callback to laptop."""
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

        # Send callback to laptop with label
        if has_label:
            self._send_callback_to_laptop(request_id, target)
        else:
            print(
                f"[VOICE@{request_id}] Skipping callback because no label was extracted"
            )

        return response_payload, has_label

    def _send_callback_to_laptop(self, request_id: str, target) -> None:
        """Send label callback to laptop via HTTP POST."""
        if not self.laptop_callback_url:
            print(f"[VOICE@{request_id}] No laptop callback URL configured, skipping")
            return

        callback_payload = {
            "request_id": request_id,
            "target": {
                "label": target.label,
                "normalized_label": target.normalized_label,
                "confidence": target.confidence,
                "reason": target.reason,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

        body = json.dumps(callback_payload).encode("utf-8")
        req = Request(
            self.laptop_callback_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "X-Request-ID": request_id,
            },
            method="POST",
        )

        try:
            with urlopen(req, timeout=10) as resp:
                status = resp.status
                print(f"[VOICE@{request_id}] Callback sent to laptop, status={status}")
        except (HTTPError, URLError) as exc:
            print(f"[VOICE@{request_id}] Callback to laptop failed: {exc}")

    def _extract_audio_from_request(self, request_id: str) -> Path:
        data = request.get_data(cache=False)
        if not data:
            raise ValueError("Audio body is empty")

        file_name = f"audio_{request_id}.wav"
        return self._write_audio(file_name, data)

    def _write_audio(self, file_name: str, audio_bytes: bytes) -> Path:
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
        "--laptop-callback-url",
        default="",
        help="Laptop callback endpoint URL (optional; leave empty when laptop is not publicly reachable)",
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
            laptop_callback_url=args.laptop_callback_url,
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
    if args.laptop_callback_url:
        print(f"Laptop callback:    {args.laptop_callback_url}")
    else:
        print("Laptop callback:    disabled (response-only mode)")
    print(f"Health check:       http://{args.server_host}:{args.server_port}/health")
    print("Press Ctrl + C để thoát.")
    print("=" * 70)

    try:
        node.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
