# AI Glasses - Camera Pipeline + Web View

## Minimal Structure

```text
glasses/
  main.py                  # Run app (start web server)
  web/
    app.py                 # Flask app + MJPEG stream routes
  vision/
    settings.py            # Stream/model/window configuration
    camera.py              # Open camera stream
    processor.py           # YOLO processing for each frame
    pipeline.py            # Combine camera + processor
  requirements.txt
```

## How It Works

1. Open camera stream from URL.
2. Thread 1 reads frames from camera into a queue.
3. Thread 2 takes frames from queue and runs YOLO.
4. Optional hand tracking runs in a separate thread.
5. Flask serves processed frames as MJPEG for browser viewing.
6. FPS is logged every second in the processing thread.

## Run

```bash
pip install -r requirements.txt
python main.py
```

Open browser: `http://127.0.0.1:5000`

Stop server with `Ctrl + C`.

Optional host/port:

```bash
python main.py --host 0.0.0.0 --port 5000
```
