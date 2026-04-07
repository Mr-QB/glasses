from __future__ import annotations

from typing import Iterator

import cv2
from flask import Flask, Response

from vision.pipeline import VisionPipeline
from voice.http_server import VoiceHTTPServer


def _mjpeg_stream(pipeline: VisionPipeline) -> Iterator[bytes]:
    while True:
        frame = pipeline.next_frame()
        if frame is None:
            break

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        jpg_bytes = encoded.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n")


def create_app(
    pipeline: VisionPipeline,
    voice_server: VoiceHTTPServer | None = None,
) -> Flask:
    app = Flask(__name__)

    @app.route("/")
    def index() -> str:
        return """
<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>AI Glasses Camera</title>
    <style>
      body { margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }
      .wrap { min-height: 100vh; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 12px; }
      img { width: min(95vw, 960px); height: auto; border: 1px solid #333; border-radius: 8px; }
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <h2>AI Glasses Live Camera</h2>
      <img src=\"/stream\" alt=\"camera stream\" />
      <p>Nhấn Ctrl + C trong terminal để dừng server.</p>
    </div>
  </body>
</html>
"""

    @app.route("/stream")
    def stream() -> Response:
        # When a client subscribes to MJPEG, keep vision active so frames are available.
        pipeline.activate(keepalive_seconds=0.0)
        return Response(
            _mjpeg_stream(pipeline),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    if voice_server is not None:
        voice_server.register_routes(app)

    return app
