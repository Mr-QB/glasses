import cv2


def open_camera(
    stream_url: str, frame_width: int, frame_height: int
) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    if not cap.isOpened():
        raise RuntimeError(f"Error. Not opened: {stream_url}")

    return cap
