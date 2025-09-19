import os
import cv2
import threading
import time
import logging
from ..config import RTSP_URL

logger = logging.getLogger("uvicorn")


def get_frame(transfrom_func=None):
    """
    最新の1フレームをJPEGにして返す
    """
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        return None

    # 古いフレームを捨てて最新を取得
    for _ in range(5):
        success, frame = cap.read()
    cap.release()

    if not success:
        return None

    # NOTE: 画像処理を行う場合はここで実施
    if transfrom_func:
        frame = transfrom_func(frame)

    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return None

    return buffer.tobytes()


def frame_generator(
    stop_event: threading.Event,
    transfrom_func=None,
    max_seconds: int = 60,
):
    """
    ストリーミング用のフレームを連続で返す
    """
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("RTSPストリームを開けませんでした")

    start_time = time.time()

    try:
        # stop_eventがセットされるまでフレームを読み続ける
        while not stop_event.is_set():
            # 最大時間を超えたら終了
            if max_seconds and (time.time() - start_time > max_seconds):
                logger.info(
                    "Streaming time exceeded %d seconds, closing stream", max_seconds
                )
                break

            # 最新フレームのみを取得
            for _ in range(5):
                success, frame = cap.read()
            if not success:
                continue

            # NOTE: 画像処理を行う場合はここで実施
            if transfrom_func:
                frame = transfrom_func(frame)

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
