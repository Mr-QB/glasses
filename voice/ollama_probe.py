from __future__ import annotations

import argparse
import json
from urllib.error import HTTPError
from urllib.error import URLError
from urllib.request import Request
from urllib.request import urlopen

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
DEFAULT_MODEL = "gemma3:4b"


def probe_ollama_api(model: str, url: str, timeout_seconds: float = 10.0) -> dict:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Reply with a single JSON object only. "
                    "No markdown, no code fences, no explanation."
                ),
            },
            {
                "role": "user",
                "content": 'Return exactly: {"ok": true, "message": "pong"}',
            },
        ],
        "options": {
            "temperature": 0.0,
            "num_predict": 256,
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
    done_reason = None
    eval_count = None
    thinking = ""
    content = ""
    if isinstance(data, dict):
        done_reason = data.get("done_reason")
        eval_count = data.get("eval_count")
        message = data.get("message", {})
        if isinstance(message, dict):
            content = str(message.get("content", "")).strip()
            thinking = str(message.get("thinking", "")).strip()

    ok = bool(content)
    if not ok and done_reason == "length":
        reason = "Model output was truncated (done_reason=length). Increase num_predict or use a non-thinking model."
    elif not ok and thinking:
        reason = "Model produced thinking but empty content."
    elif not ok:
        reason = "Empty message.content returned by Ollama."
    else:
        reason = None

    return {
        "ok": ok,
        "reason": reason,
        "done_reason": done_reason,
        "eval_count": eval_count,
        "raw_response": raw,
        "message_content": content,
        "message_thinking": thinking,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe Ollama chat API and verify it returns a response",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    parser.add_argument(
        "--url",
        default=DEFAULT_OLLAMA_URL,
        help="Ollama chat API URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds",
    )
    args = parser.parse_args()

    try:
        result = probe_ollama_api(args.model, args.url, args.timeout)
    except HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP error {exc.code}: {error_text}") from exc
    except URLError as exc:
        raise SystemExit(f"Cannot reach Ollama API: {exc}") from exc
    except Exception as exc:
        raise SystemExit(f"Failed to probe Ollama: {exc}") from exc

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
