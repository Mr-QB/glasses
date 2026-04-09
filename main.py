"""
DEPRECATED: Unified architecture

This file is kept for reference. Use instead:
  - main_laptop.py: Vision pipeline + audio gateway (runs on laptop)
  - main_voice_server.py: STT + Ollama processing (runs on server)

See ARCHITECTURE.md for distributed 2-node setup.
"""

import argparse

from web.app import create_app
from vision.pipeline import VisionPipeline
from vision.settings import VisionSettings
from shared.target_handoff import TargetHandoffBus
from voice.assistant import VoiceAssistant
from voice.http_server import VoiceHTTPServer
from voice.settings import STTSettings
from voice.settings import VoiceSettings


def main():
    parser = argparse.ArgumentParser(
        description="AI Glasses camera preview (LEGACY - use main_laptop.py or main_voice_server.py)"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Flask host")
    parser.add_argument("--port", type=int, default=5051, help="Flask port")
    args = parser.parse_args()

    try:
        settings = VisionSettings(yolo_device="cuda:0")
        pipeline = VisionPipeline(settings)
        target_bus = TargetHandoffBus()
        pipeline.attach_target_bus(target_bus)
        voice_assistant = VoiceAssistant(
            settings=VoiceSettings(enabled=False),
            stt_settings=STTSettings(device_preference="cuda"),
        )
        voice_server = VoiceHTTPServer(
            assistant=voice_assistant,
            target_bus=target_bus,
            ollama_timeout_seconds=12.0,
            ollama_max_output_tokens=80,
            on_audio_received=lambda: pipeline.activate(
                settings.active_keepalive_seconds
            ),
        )
    except RuntimeError as exc:
        print(exc)
        return

    pipeline.start()
    app = create_app(pipeline, voice_server=voice_server)

    print("Unified Flask runtime started.")
    print(
        "Vision pipeline is idle by default and activates on /transcribe audio requests."
    )
    print("CUDA mode forced: YOLOE device=cuda:0, STT device preference=cuda.")
    print(f"Open stream: http://{args.host}:{args.port}/")
    print(f"Audio POST:  http://{args.host}:{args.port}/transcribe")
    print("Press Ctrl + C để thoát.")

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
