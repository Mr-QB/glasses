from __future__ import annotations

import argparse

from voice.http_server import VoiceHTTPServer
from voice.settings import STTSettings
from voice.stt import PhoWhisperSTT


def main() -> None:
    parser = argparse.ArgumentParser(description="PhoWhisper Vietnamese STT CLI")
    parser.add_argument("--audio", help="Path to audio file (wav/mp3/flac/m4a)")
    parser.add_argument(
        "--mic-seconds",
        type=float,
        default=0.0,
        help="Record from microphone for N seconds (if > 0)",
    )
    parser.add_argument(
        "--model-id",
        default="vinai/PhoWhisper-large",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--http-server",
        action="store_true",
        help="Run Flask HTTP server for remote audio transcription",
    )
    parser.add_argument("--http-host", default="0.0.0.0", help="HTTP host")
    parser.add_argument("--http-port", type=int, default=5052, help="HTTP port")
    args = parser.parse_args()

    if args.http_server:
        server = VoiceHTTPServer(host=args.http_host, port=args.http_port)
        server.serve_forever()
        return

    if not args.audio and args.mic_seconds <= 0:
        parser.error("Provide --audio or set --mic-seconds > 0")

    settings = STTSettings(model_id=args.model_id)
    stt = PhoWhisperSTT(settings)

    if args.audio:
        result = stt.transcribe_file(args.audio)
    else:
        result = stt.transcribe_microphone(seconds=args.mic_seconds)

    print("\n--- STT Result ---")
    print(result.text)


if __name__ == "__main__":
    main()
