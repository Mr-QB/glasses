import argparse

from web import create_app
from vision.pipeline import VisionPipeline
from vision.settings import VisionSettings


def main():
    parser = argparse.ArgumentParser(description="AI Glasses web camera stream")
    parser.add_argument("--host", default="0.0.0.0", help="Web server host")
    parser.add_argument("--port", type=int, default=5000, help="Web server port")
    args = parser.parse_args()
    
    try:
        settings = VisionSettings()
        pipeline = VisionPipeline(settings)
    except RuntimeError as exc:
        print(exc)
        return

    pipeline.start()
    app = create_app(pipeline)

    print(f"Web camera running at: http://{args.host}:{args.port}")
    print("Press Ctrl + C to stop.")

    try:
        app.run(
            host=args.host,
            port=args.port,
            debug=False,
            use_reloader=False,
            threaded=True,
        )
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
