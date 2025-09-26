import cv2
import threading
import time
import logging
from typing import Callable, Generator, Optional, Dict
from ..config import RTSP_URL

logger = logging.getLogger("uvicorn")


class CameraFrame():
    def __init__(self, rtsp_url: str=RTSP_URL):
        # カメラ接続用
        self.rtsp_url = rtsp_url  
        
        # 動体検知用の状態
        self.is_motion = False
        self.last_motion_time = None
        self.prev_frame = None
        self.prev_frame_time = 0


    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        """
        RTSPストリームを開き、バッファサイズを設定して返す。
        開けなければNoneを返す。
        """
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            return None
        return cap


    def _read_latest_frame(self, cap: cv2.VideoCapture) -> Optional[cv2.Mat]:
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


    def get_features(self, extract_func):
        """
        最新の1フレームを取得し、extract_funcで特徴抽出を行い返す。
        """
        cap = self._open_capture()
        if cap is None:
            return None

        frame = self._read_latest_frame(cap)
        cap.release()

        if frame is None:
            return None

        features = extract_func(frame)
        return features


    def get_frame(self, transform_func=None):
        """
        最新の1フレームをJPEGエンコードして返す。
        transform_funcが指定されていればフレームに適用する。
        """
        cap = self._open_capture()
        if cap is None:
            return None

        frame = self._read_latest_frame(cap)
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
        self,
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
        cap = self._open_capture()
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

                frame = self._read_latest_frame(cap)
                if frame is None:
                    continue

                # 5秒おきにprev_frameを更新
                current_time = time.time()
                if self.prev_frame is None or (current_time - self.prev_frame_time) >= 5:
                    self.prev_frame = frame.copy()
                    self.prev_frame_time = current_time
                    self.is_motion = False
                else:
                    # 現在のフレームとprev_frameを白黒化、ぼかし
                    prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
                    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
                    curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

                    # フレーム間の差分抽出
                    frame_delta = cv2.absdiff(prev_gray, curr_gray)
                    _, thresh_img = cv2.threshold(frame_delta, 50, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for cnt in contours:
                        if cv2.contourArea(cnt) > 5000:
                            self.is_motion = True
                            break

                    if self.is_motion:
                        self.last_motion_time = current_time

                    print("", self.is_motion, self.last_motion_time)
                
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
            
            # リセット
            self.prev_frame = None
            self.prev_frame_time = 0
            self.is_motion = False
            self.last_motion_time = None
