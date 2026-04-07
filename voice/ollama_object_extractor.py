from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen


DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"


def build_prompt(transcript: str) -> str:
    return (
        "Bạn là bộ trích xuất ý định từ câu tiếng Việt.\n"
        "Nhiệm vụ: xác định chính xác đồ vật mà người dùng cần lấy từ câu STT.\n"
        "normalized_object phải là tên ngắn gọn phù hợp cho vision model, ưu tiên tiếng Anh nếu có thể.\n"
        "Chỉ trả về JSON hợp lệ, không thêm giải thích ngoài JSON.\n"
        "Schema:\n"
        "{\n"
        '  "object": "tên đồ vật ngắn gọn hoặc null",\n'
        '  "normalized_object": "tên chuẩn hoá ngắn gọn hoặc null",\n'
        '  "confidence": 0.0,\n'
        '  "reason": "lý do ngắn gọn"\n'
        "}\n"
        "Nếu không xác định được đồ vật thì object và normalized_object phải là null.\n"
        f"Câu STT: {transcript}"
    )


def parse_ollama_output(raw_text: str) -> dict:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "object": None,
            "normalized_object": None,
            "confidence": 0.0,
            "reason": "Ollama did not return valid JSON",
            "raw_output": raw_text,
        }


def call_ollama(
    transcript: str,
    model: str,
    url: str,
    timeout_seconds: float = 20.0,
    max_output_tokens: int = 96,
) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract the object the user needs to take from a Vietnamese transcript. "
                    "Return JSON only."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(transcript),
            },
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": int(max_output_tokens),
        },
    }

    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=max(1.0, float(timeout_seconds))) as response:
        raw = response.read().decode("utf-8", errors="replace")

    data = json.loads(raw)
    content = data["message"]["content"].strip()
    return parse_ollama_output(content)


def read_transcript(args: argparse.Namespace) -> str:
    if args.text:
        return args.text.strip()

    if args.transcript_file:
        return args.transcript_file.read_text(encoding="utf-8").strip()

    return sys.stdin.read().strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call local Ollama to extract the object the user needs to take from STT text",
    )
    parser.add_argument("--text", help="Transcript text from STT")
    parser.add_argument(
        "--transcript-file",
        type=Path,
        help="Read transcript text from a UTF-8 text file",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model name",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama chat API URL",
    )
    args = parser.parse_args()

    transcript = read_transcript(args)
    if not transcript:
        parser.error("Provide --text, --transcript-file, or pipe transcript via stdin")

    try:
        result = call_ollama(transcript, args.model, args.url)
    except HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP error {exc.code}: {error_text}") from exc
    except URLError as exc:
        raise SystemExit(f"Cannot reach Ollama API: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Failed to call Ollama: {exc}") from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
