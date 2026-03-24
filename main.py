import argparse

from web.app import create_app
from vision.pipeline import VisionPipeline
from vision.settings import VisionSettings


def main():
    parser = argparse.ArgumentParser(description="AI Glasses camera preview")
    parser.add_argument("--host", default="0.0.0.0", help="Flask host")
    parser.add_argument("--port", type=int, default=5051, help="Flask port")
    args = parser.parse_args()

    try:
        settings = VisionSettings()
        pipeline = VisionPipeline(settings)
    except RuntimeError as exc:
        print(exc)
        return

    pipeline.start()
    app = create_app(pipeline)

    print("Flask MJPEG stream started.")
    print(f"Open: http://{args.host}:{args.port}")
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
