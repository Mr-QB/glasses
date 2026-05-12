# AI Glasses - Distributed Vision + Voice Processing

## Architecture Overview

AI Glasses now supports a **distributed 2-node architecture**:
- **Laptop Node** (`main_laptop.py`): Runs vision pipeline with MJPEG stream, audio ingress gateway
- **Voice Server Node** (`main_voice_server.py`): Runs STT + Ollama for object extraction, sends labels back to laptop

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design, data contracts, and deployment scenarios.

## Project Structure

```text
glasses/
  main_laptop.py           # Laptop node: vision + audio gateway + callback handler
  main_voice_server.py     # Voice server node: STT + Ollama + label callback
  main.py                  # (Legacy) Unified app for reference
  shared/
    target_handoff.py      # TargetHandoff data class for label passing
  web/
    app.py                 # Flask web app template (vision stream routes)
  vision/
    settings.py            # Vision pipeline config
    camera.py              # Camera stream ingestion
    processor.py           # YOLO object detection
    pipeline.py            # Vision pipeline orchestration
  voice/
    settings.py            # STT + Ollama configuration
    http_server.py         # HTTP server for voice processing (legacy)
    stt.py                 # Speech-to-text using PhoWhisper
    assistant.py           # Voice assistant orchestration
    ollama_object_extractor.py  # Object extraction via Ollama
  requirements.txt         # Python dependencies
  ARCHITECTURE.md          # 2-node architecture documentation
```

## Quick Start: Distributed 2-Node Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run Voice Server (Remote Machine or Port 5052)

```bash
python main_voice_server.py \
  --server-host 0.0.0.0 \
  --server-port 5052 \
  --laptop-callback-url http://laptop-ip:5051/target_callback
```

Voice server listens on:
- `POST /transcribe` - Receive WAV, return STT result + label
- `GET /health` - Health check

### Run Laptop Node (Local Machine, Port 5051)

```bash
python main_laptop.py \
  --laptop-host 0.0.0.0 \
  --laptop-port 5051 \
  --remote-voice-url http://voice-server-ip:5052
```

Laptop node listens on:
- `GET /` - Video MJPEG stream (open in browser: `http://127.0.0.1:5051`)
- `POST /audio` - Audio ingress from ESP32 (gateway to voice server)
- `POST /target_callback` - Label callback from voice server (drives vision activation)
- `GET /health` - Health check

### Send Audio from ESP32/Client

```bash
curl -X POST http://laptop-ip:5051/audio \
  -H "Content-Type: audio/wav" \
  --data-binary @sample.wav
```

## Testing

### Local Development (All-in-one Machine)

```bash
# Terminal 1: Start voice server
python main_voice_server.py

# Terminal 2: Start laptop node
python main_laptop.py

# Terminal 3: Send test audio
python test_post_encoded_mp3.py --api-url http://127.0.0.1:5051/audio
```

### With ESP32-CAM

Configure ESP32-CAM to:
1. Stream MJPEG/RTSP video to laptop (URL in `vision/settings.py`)
2. POST WAV audio to `http://laptop-ip:5051/audio`

See [ARCHITECTURE.md](ARCHITECTURE.md) for ESP32 code example.

## Utilities

Extract object from transcript using Ollama (standalone):

```bash
python -m voice.ollama_object_extractor --text "lấy giúp tôi cái cốc ở trên bàn"
```
