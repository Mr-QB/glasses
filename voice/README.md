# Voice Module (Standalone)

Module `voice/` is intentionally separated from the main vision pipeline.

## Features

- Vietnamese STT with `vinai/PhoWhisper-large`
- File transcription (`wav`, `mp3`, `flac`, `m4a`, ...)
- Microphone recording + transcription
- Optional TTS worker (existing module)

## Install

```bash
pip install -r voice/requirements.txt
```

## Quick Usage

Transcribe an audio file:

```bash
python -m voice.cli --audio path/to/sample.wav
```

Record from microphone for 6 seconds and transcribe:

```bash
python -m voice.cli --mic-seconds 6
```

Run HTTP server:

```bash
python -m voice.cli --http-server --http-host 0.0.0.0 --http-port 5052
```

Run the unified app that streams video and accepts audio on the same process:

```bash
python main.py --host 0.0.0.0 --port 5051
```

## HTTP API

Base URL: `http://<host>:5052`

Health check:

```bash
curl http://127.0.0.1:5052/health
```

Transcribe multipart file upload:

```bash
curl -X POST http://127.0.0.1:5052/transcribe \
	-F "audio=@raw_data/sample.m4a"
```

Transcribe raw binary body:

```bash
curl -X POST http://127.0.0.1:5052/transcribe \
	-H "Content-Type: audio/mp4" \
	-H "X-Filename: sample.m4a" \
	--data-binary "@raw_data/sample.m4a"
```

Transcribe JSON base64 payload:

```json
{
	"filename": "sample.wav",
	"audio_base64": "<base64>"
}
```

Server response:

```json
{"text":"...","language":"vi","source":"..."}
```

Extract the object user wants to take from an STT transcript using local Ollama:

```bash
python -m voice.ollama_object_extractor --text "lấy giúp tôi cái cốc ở trên bàn"
```

Default Ollama model for the integrated flow:

```text
qwen3.5:4b
```

Or read transcript text from a file:

```bash
python -m voice.ollama_object_extractor --transcript-file transcript.txt
```

## Programmatic Usage

```python
from voice.settings import STTSettings
from voice.stt import PhoWhisperSTT

stt = PhoWhisperSTT(STTSettings(model_id="vinai/PhoWhisper-large"))
result = stt.transcribe_file("sample.wav")
print(result.text)
```

## Notes

- First run will download model weights from Hugging Face.
- GPU is used automatically when available; otherwise CPU is used.
