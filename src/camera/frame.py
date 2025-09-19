import cv2
import threading
import time
import logging
from typing import Callable, Generator, Optional, Dict
from ..config import RTSP_URL

logger = logging.getLogger("uvicorn")


def _open_capture() -> Optional[cv2.VideoCapture]:
    """
    RTSPストリームを開き、バッファサイズを設定して返す。
    開けなければNoneを返す。
    """
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        return None
    return cap


def _read_latest_frame(cap: cv2.VideoCapture) -> Optional[cv2.Mat]:
    """
    古いフレームを捨てて最新のフレームを取得する。
    取得できなければNoneを返す。
    """
    success = False
    frame = None
    for _ in range(5):
        success, frame = cap.read()
    if not success:
        return None
    return frame


def get_features(extract_func: Callable[[cv2.Mat], any]) -> Optional[Dict]:
    """
    最新の1フレームを取得し、extract_funcで特徴抽出を行い返す。
    """
    cap = _open_capture()
    if cap is None:
        return None

    frame = _read_latest_frame(cap)
    cap.release()

    if frame is None:
        return None

    features = extract_func(frame)
    return features


def get_frame(
    transform_func: Optional[Callable[[cv2.Mat], cv2.Mat]] = None,
) -> Optional[bytes]:
    """
    最新の1フレームをJPEGエンコードして返す。
    transform_funcが指定されていればフレームに適用する。
    """
    cap = _open_capture()
    if cap is None:
        return None

    frame = _read_latest_frame(cap)
    cap.release()

    if frame is None:
        return None

    if transform_func:
        frame = transform_func(frame)

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None

    return buffer.tobytes()


def frame_generator(
    stop_event: threading.Event,
    transform_func: Optional[Callable[[cv2.Mat], cv2.Mat]] = None,
    max_seconds: int = 60,
) -> Generator[bytes, None, None]:
    """
    ストリーミング用のフレームを連続で返すジェネレータ。
    stop_eventがセットされるまで、またはmax_secondsを超えるまでフレームを取得し続ける。
    transform_funcが指定されていればフレームに適用する。
    """
    cap = _open_capture()
    if cap is None:
        raise RuntimeError("RTSPストリームを開けませんでした")

    start_time = time.time()

    try:
        while not stop_event.is_set():
            if max_seconds and (time.time() - start_time > max_seconds):
                logger.info(
                    "Streaming time exceeded %d seconds, closing stream", max_seconds
                )
                break

            frame = _read_latest_frame(cap)
            if frame is None:
                continue

            if transform_func:
                frame = transform_func(frame)

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()
        logger.info("RTSP connection closed (generator finished)")
