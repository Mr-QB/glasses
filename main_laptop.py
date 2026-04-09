"""
AI Glasses Laptop Node - Vision Pipeline + Audio Gateway

Runs locally on laptop:
  - Streams MJPEG video from camera to browser
  - Receives WAV audio from ESP32 via /audio endpoint
  - Forwards audio to remote voice server
    - Activates vision pipeline from remote response payload

Usage:
    python main_laptop.py \\n    --laptop-host 0.0.0.0 \\n    --laptop-port 5052 \\n    --remote-voice-url https://voice.uet707e3.site

Endpoints:
  GET  /                - Video MJPEG stream
  POST /audio           - Audio ingress from ESP32 (WAV bytes)
  GET  /health          - Health check
"""

import argparse
import http.client
import json
import socket
import uuid
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

try:
    import paho.mqtt.client as mqtt_client
except Exception:
    mqtt_client = None

from web.app import create_app
from vision.pipeline import VisionPipeline
from vision.settings import VisionSettings
from shared.target_handoff import TargetHandoffBus, TargetHandoff


class MQTTGuidePublisher:
    """Publish guidance ID updates to MQTT."""

    def __init__(self, host: str, port: int, topic: str) -> None:
        self.host = host
        self.port = int(port)
        self.topic = topic
        self._mqtt_mod = mqtt_client
        self._client = None
        self._enabled = False

        if self._mqtt_mod is None:
            print("[MQTT] paho-mqtt is not installed. MQTT publish disabled.")
            return

        client_id = f"glasses-guide-{uuid.uuid4().hex[:8]}"
        self._client = self._mqtt_mod.Client(
            client_id=client_id,
            protocol=self._mqtt_mod.MQTTv311,
        )
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect

    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            print(
                f"[MQTT] Connected broker {self.host}:{self.port} | topic={self.topic}"
            )
            return
        print(f"[MQTT] Connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        if rc == 0:
            print("[MQTT] Disconnected cleanly.")
            return
        print(f"[MQTT] Disconnected unexpectedly rc={rc}. Reconnect will retry.")

    def start(self) -> None:
        if self._client is None:
            return
        try:
            self._client.connect_async(self.host, self.port, keepalive=30)
            self._client.loop_start()
            self._enabled = True
            print(
                f"[MQTT] Publisher started | broker={self.host}:{self.port} | topic={self.topic}"
            )
        except Exception as exc:
            print(f"[MQTT] Failed to start publisher: {exc}")
            self._enabled = False

    def publish_guide(self, guide_id: str) -> None:
        if not self._enabled or self._client is None:
            return

        payload = str(guide_id).strip().lower()
        if not payload:
            return

        try:
            info = self._client.publish(
                self.topic,
                payload=payload,
                qos=0,
                retain=False,
            )
            if (
                self._mqtt_mod is not None
                and getattr(info, "rc", self._mqtt_mod.MQTT_ERR_SUCCESS)
                != self._mqtt_mod.MQTT_ERR_SUCCESS
            ):
                print(f"[MQTT] Publish failed rc={getattr(info, 'rc', None)}")
        except Exception as exc:
            print(f"[MQTT] Publish exception: {exc}")

    def close(self) -> None:
        if not self._enabled or self._client is None:
            return

        self._enabled = False
        try:
            self._client.disconnect()
        except Exception:
            pass
        try:
            self._client.loop_stop()
        except Exception:
            pass


class LaptopNode:
    """Laptop vision node that gateways audio to remote voice server."""

    def __init__(
        self,
        laptop_host: str = "0.0.0.0",
        laptop_port: int = 5052,
        remote_voice_url: str = "https://voice.uet707e3.site",
        mqtt_host: str = "14.246.4.81",
        mqtt_port: int = 1883,
        mqtt_topic: str = "guide/guide_hand",
    ):
        self.laptop_host = laptop_host
        self.laptop_port = laptop_port
        self.remote_voice_url = remote_voice_url.rstrip("/")
        self.mqtt_host = mqtt_host
        self.mqtt_port = mqtt_port
        self.mqtt_topic = mqtt_topic

        # Local vision pipeline
        self.settings = VisionSettings(yolo_device="cuda:0")
        self.guide_publisher = MQTTGuidePublisher(
            host=self.mqtt_host,
            port=self.mqtt_port,
            topic=self.mqtt_topic,
        )
        self.guide_publisher.start()
        self.pipeline = VisionPipeline(
            self.settings,
            on_guidance_change=self.guide_publisher.publish_guide,
        )
        self.target_bus = TargetHandoffBus()
        self.pipeline.attach_target_bus(self.target_bus)

    def close(self) -> None:
        self.pipeline.stop()
        self.guide_publisher.close()

    def setup_app(self):
        """Create Flask app with vision stream + audio ingress."""
        app = create_app(self.pipeline, voice_server=None)

        @app.post("/audio")
        def receive_audio() -> tuple[dict, int]:
            """Receive WAV audio from ESP32, forward to remote voice server."""
            try:
                audio_bytes = self._extract_audio_body()
                request_id = str(uuid.uuid4())

                # Forward to remote voice server
                status, response = self._forward_to_remote_voice(
                    audio_bytes, request_id
                )

                remote_payload: dict[str, Any] | None = None
                try:
                    remote_payload = json.loads(response)
                except Exception:
                    remote_payload = None

                print(
                    f"[LAPTOP<-VOICE@{request_id}] "
                    f"status={status} payload={remote_payload if remote_payload is not None else response!r}"
                )

                if status >= 500:
                    return {
                        "status": "error",
                        "code": "REMOTE_VOICE_ERROR",
                        "request_id": request_id,
                        "remote_status": status,
                        "message": "Remote voice server failed while processing audio",
                        "remote_response": remote_payload,
                    }, 502

                if status == 400:
                    return {
                        "status": "error",
                        "code": "INVALID_AUDIO_REQUEST",
                        "request_id": request_id,
                        "remote_status": status,
                        "message": "Remote voice server rejected request format",
                        "remote_response": remote_payload,
                    }, 400

                activated = False
                target_label: str | None = None
                if isinstance(remote_payload, dict):
                    activated, target_label = self._apply_target_from_payload(
                        remote_payload
                    )

                if status == 422:
                    self.guide_publisher.publish_guide("recall")
                    return {
                        "status": "no_label",
                        "code": "NO_LABEL_EXTRACTED",
                        "request_id": request_id,
                        "remote_status": status,
                        "activated": False,
                        "target_label": None,
                        "message": "Audio was transcribed but no reliable object label was extracted",
                        "remote_response": remote_payload,
                    }, 422

                if not activated:
                    self.guide_publisher.publish_guide("recall")

                return {
                    "status": "ok",
                    "code": "LABEL_EXTRACTED" if activated else "REMOTE_OK_NO_TARGET",
                    "request_id": request_id,
                    "remote_status": status,
                    "activated": activated,
                    "target_label": target_label,
                    "message": (
                        "Label received and vision activated"
                        if activated
                        else "Remote completed but did not return target payload"
                    ),
                    "remote_response": remote_payload,
                }, 200
            except ValueError as exc:
                return {"error": str(exc)}, 400
            except RuntimeError as exc:
                return {
                    "status": "error",
                    "code": "REMOTE_VOICE_UNREACHABLE",
                    "request_id": request_id,
                    "message": str(exc),
                    "remote_url": self.remote_voice_url,
                }, 502
            except Exception as exc:
                return {"error": f"Internal gateway error: {exc}"}, 500

        @app.get("/health")
        def health() -> dict:
            return {
                "status": "ok",
                "node": "laptop",
                "timestamp": datetime.utcnow().isoformat(),
            }

        return app

    def _extract_audio_body(self) -> bytes:
        """Extract audio bytes from request."""
        from flask import request

        data = request.get_data(cache=False)
        if not data:
            raise ValueError("Audio body is empty")
        return data

    def _forward_to_remote_voice(
        self, audio_bytes: bytes, request_id: str
    ) -> tuple[int, str]:
        """Forward audio to remote voice server, get response."""
        parsed = urlparse(self.remote_voice_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise RuntimeError(
                f"Invalid remote voice URL: {self.remote_voice_url}. "
                "Expected format: http(s)://host[:port]"
            )

        host = parsed.hostname
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        path_prefix = parsed.path.rstrip("/")
        endpoint_path = f"{path_prefix}/transcribe" if path_prefix else "/transcribe"

        headers = {
            "Host": parsed.netloc,
            "Content-Type": "audio/wav",
            "Content-Length": str(len(audio_bytes)),
            "X-Request-ID": request_id,
            "X-Filename": f"audio_{request_id}.wav",
            "Connection": "close",
        }

        connection_cls = (
            http.client.HTTPSConnection
            if parsed.scheme == "https"
            else http.client.HTTPConnection
        )
        connection = connection_cls(host, port, timeout=30)

        try:
            print(
                f"[LAPTOP->VOICE@{request_id}] "
                f"POST {parsed.scheme}://{parsed.netloc}{endpoint_path} "
                f"| bytes={len(audio_bytes)}"
            )
            connection.request(
                "POST",
                endpoint_path,
                body=audio_bytes,
                headers=headers,
            )
            response = connection.getresponse()
            status = int(response.status)
            text = response.read().decode("utf-8", errors="replace")
            print(f"[LAPTOP->VOICE@{request_id}] status={status}")
            return status, text
        except (socket.timeout, ConnectionError, OSError) as exc:
            raise RuntimeError(f"Failed to reach remote voice server: {exc}") from exc
        finally:
            connection.close()

    def _apply_target_from_payload(
        self, payload: dict[str, Any]
    ) -> tuple[bool, str | None]:
        target_data = payload.get("target") if isinstance(payload, dict) else None
        if not isinstance(target_data, dict):
            return False, None

        normalized_label = target_data.get("normalized_label")
        if not normalized_label:
            return False, None

        transcript = str(payload.get("text") or "").strip()

        target = TargetHandoff(
            label=target_data.get("label"),
            normalized_label=normalized_label,
            confidence=target_data.get("confidence", 0.0),
            transcript=transcript,
            reason=target_data.get("reason"),
            raw_output=response_text(payload),
        )
        print(
            f"[LAPTOP] Applying target from voice server: "
            f"label={target.label!r} normalized_label={target.normalized_label!r} "
            f"confidence={target.confidence:.2f}"
        )
        self.pipeline.apply_target_handoff(target)
        print(
            f"[LAPTOP] Vision pipeline activated for "
            f"{self.settings.active_keepalive_seconds:.1f}s"
        )
        return True, str(normalized_label)


def response_text(payload: dict[str, Any]) -> str | None:
    text = payload.get("text")
    if text is None:
        return None
    text_value = str(text).strip()
    return text_value or None


def main():
    parser = argparse.ArgumentParser(
        description="AI Glasses Laptop Node (Vision + Audio Gateway)"
    )
    parser.add_argument("--laptop-host", default="0.0.0.0", help="Laptop Flask host")
    parser.add_argument(
        "--laptop-port", type=int, default=5052, help="Laptop Flask port"
    )
    parser.add_argument(
        "--remote-voice-url",
        default="https://voice.uet707e3.site",
        help="Remote voice server URL",
    )
    parser.add_argument(
        "--mqtt-host",
        default="14.246.4.81",
        help="MQTT broker host",
    )
    parser.add_argument(
        "--mqtt-port",
        type=int,
        default=1883,
        help="MQTT broker port",
    )
    parser.add_argument(
        "--mqtt-topic",
        default="guide/guide_hand",
        help="MQTT topic for guidance ID updates",
    )
    args = parser.parse_args()

    try:
        node = LaptopNode(
            laptop_host=args.laptop_host,
            laptop_port=args.laptop_port,
            remote_voice_url=args.remote_voice_url,
            mqtt_host=args.mqtt_host,
            mqtt_port=args.mqtt_port,
            mqtt_topic=args.mqtt_topic,
        )
    except RuntimeError as exc:
        print(f"Failed to initialize laptop node: {exc}")
        return

    node.pipeline.start()
    app = node.setup_app()

    print("=" * 70)
    print("🖥️  LAPTOP NODE STARTED")
    print("=" * 70)
    print(f"Vision stream:      http://{args.laptop_host}:{args.laptop_port}/")
    print(
        f"Audio ingress:      http://{args.laptop_host}:{args.laptop_port}/audio (POST)"
    )
    print(f"Remote voice:       {args.remote_voice_url}")
    print(f"MQTT guide topic:   mqtt://{args.mqtt_host}:{args.mqtt_port}/{args.mqtt_topic}")
    print(f"Health check:       http://{args.laptop_host}:{args.laptop_port}/health")
    print("Press Ctrl + C để thoát.")
    print("=" * 70)

    try:
        app.run(
            host=args.laptop_host,
            port=args.laptop_port,
            debug=False,
            threaded=True,
            use_reloader=False,
        )
    finally:
        node.close()


if __name__ == "__main__":
    main()
