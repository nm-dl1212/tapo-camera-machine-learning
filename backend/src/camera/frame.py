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


def get_features(extract_func):
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


def get_frame(transform_func=None):
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


is_motion = False
last_motion_time = None
_prev_frame = None
_prev_frame_time = 0

def frame_generator(
    stop_event: threading.Event,
    transform_func: Optional[Callable[[cv2.Mat], cv2.Mat]] = None,
    max_seconds: int = 180,
) -> Generator[bytes, None, None]:
    """
    ストリーミング用のフレームを連続で返すジェネレータ。
    stop_eventがセットされるまで、またはmax_secondsを超えるまでフレームを取得し続ける。
    transform_funcが指定されていればフレームに適用する。
    5秒おきにprev_frameを格納し、現在のフレームと差があるときのみis_motionフラグをTrueにする。
    """
    global is_motion, _prev_frame, _prev_frame_time, last_motion_time

    cap = _open_capture()
    if cap is None:
        raise RuntimeError("RTSPストリームを開けませんでした")

    start_time = time.time()
    _prev_frame = None
    _prev_frame_time = 0
    is_motion = False
    last_motion_time = None

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

            # 5秒おきにprev_frameを更新
            current_time = time.time()
            if _prev_frame is None or (current_time - _prev_frame_time) >= 5:
                _prev_frame = frame.copy()
                _prev_frame_time = current_time
                is_motion = False
            else:
                # 現在のフレームとprev_frameを白黒化、ぼかし
                prev_gray = cv2.cvtColor(_prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
                curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

                # フレーム間の差分抽出
                frame_delta = cv2.absdiff(prev_gray, curr_gray)
                _, thresh_img = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv2.contourArea(cnt) > 5000:
                        is_motion = True
                        break

                if is_motion:
                    last_motion_time = current_time

                print("", is_motion, last_motion_time)
            
            # リアルタイムの画像変換
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


def motion_generator_with_motion(
    stop_event: threading.Event,
    transform_func: Optional[Callable[[cv2.Mat], cv2.Mat]] = None,
    threshold: int = 50,
    min_area: int = 5000,
    max_seconds: int = 180,
) -> Generator[bytes, None, None]:
    """
    動体検知付きのフレームジェネレーター
    - transform_func: フレームに追加で適用する関数
    - threshold: ピクセル差の閾値
    - min_area: 動きとみなす最小領域
    - max_seconds: 最大ストリーミング時間
    """
    cap = _open_capture()
    if cap is None:
        raise RuntimeError("RTSPストリームを開けませんでした")

    start_time = time.time()
    prev_frame: Optional[cv2.Mat] = None

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

            frame_copy = frame.copy()

            # 動体検知
            if prev_frame is not None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
                prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
                curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

                frame_delta = cv2.absdiff(prev_gray, curr_gray)
                _, thresh_img = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for cnt in contours:
                    if cv2.contourArea(cnt) < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)

            prev_frame = frame.copy()

            # 追加の変換処理
            if transform_func:
                frame_copy = transform_func(frame_copy)

            ret, buffer = cv2.imencode(".jpg", frame_copy)
            if not ret:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        cap.release()
        logger.info("RTSP connection closed (motion_generator_with_motion finished)")
