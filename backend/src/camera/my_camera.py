import cv2
import threading
import time
import logging
from onvif import ONVIFCamera
from typing import Callable, Generator, Optional, Dict

logger = logging.getLogger("uvicorn")


class MyCamera:
    def __init__(
        self,
        ip_address: str,
        username: str,
        password: str,
        port: int = 554,
        stream_path: str = "stream1",
        onvif_port: int = 2020,
    ):
        # カメラ接続用
        self.ip_address = ip_address
        self.username = username
        self.password = password
        self.port = port
        self.stream = stream_path
        self.onvif_port = onvif_port

        # urlを組み立て
        self.rtsp_url = (
            f"rtsp://{username}:{password}@{ip_address}:{port}/{stream_path}"
        )

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
        cap = cv2.VideoCapture(self.rtsp_url)
        # cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
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

    def get_frame(self, transform_func=None, extract_func=None):
        """
        最新の1フレームをJPEGエンコードして返す。
        transform_funcが指定されていればフレームに適用する。
        """
        cap = self._open_capture()
        if cap is None:
            return None, None

        frame = self._read_latest_frame(cap)
        cap.release()

        if frame is None:
            return None, None

        # 画像変換
        if transform_func:
            frame = transform_func(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            return None

        # 特徴抽出
        features = None
        if extract_func:
            features = extract_func(frame)

        return buffer.tobytes(), features

    def frame_generator(
        self,
        stop_event: threading.Event,
        enable_motion_detection: bool = False,
        transform_func: Optional[Callable[[cv2.Mat], cv2.Mat]] = None,
        max_seconds: int = 604800,
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
            while not stop_event.is_set(): ## stop_eventが贈られるまで、この中をループする

                # 最大時間を超えたら終了
                if max_seconds and (time.time() - start_time > max_seconds):
                    logger.info(
                        "Streaming time exceeded %d seconds, closing stream",
                        max_seconds,
                    )
                    break

                frame = self._read_latest_frame(cap)
                if frame is None:
                    continue
                

                if self.prev_frame is None:
                    # 初期化
                    self.prev_frame = frame.copy()
                    self.prev_frame_time = time.time()
                    continue
                
                # 動体検知
                if enable_motion_detection:
                    current_time = time.time()
                    if (current_time - self.prev_frame_time) >= 5:
                        # 現在のフレームとprev_frameを白黒化、ぼかし
                        prev_gray = cv2.cvtColor(self.prev_frame, cv2.COLOR_BGR2GRAY)
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
                        curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

                        # フレーム間の差分抽出
                        frame_delta = cv2.absdiff(prev_gray, curr_gray)
                        _, thresh_img = cv2.threshold(
                            frame_delta, 50, 255, cv2.THRESH_BINARY
                        )
                        contours, _ = cv2.findContours(
                            thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                        )
                        
                        # 十分な大きさの動体があれば、is_motionをTrueに
                        self.is_motion = False
                        for cnt in contours:
                            if cv2.contourArea(cnt) > 5000:
                                self.is_motion = True
                                break

                        if self.is_motion:
                            self.last_motion_time = current_time

                        # 最後に時間とフレームをを更新する
                        self.prev_frame = frame.copy()
                        self.prev_frame_time = current_time

                # リアルタイムの画像変換
                if transform_func:
                    frame = transform_func(frame)

                ret, buffer = cv2.imencode(".jpg", frame)
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()
                yield (
                    b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                    + frame_bytes
                    + b"\r\n"
                )
        finally:
            cap.release()
            logger.info("RTSP connection closed (generator finished)")

            # リセット
            self.prev_frame = None
            self.prev_frame_time = 0
            self.is_motion = False
            self.last_motion_time = None

    """
    以下、PTZ制御用の関数
    """

    def _connect_onvif_camera(self) -> ONVIFCamera:
        """
        ONVIFカメラに接続し、ONVIFCameraオブジェクトを返す。
        接続できなければRuntimeErrorを発生させる。
        """
        mycam = ONVIFCamera(
            self.ip_address, self.onvif_port, self.username, self.password
        )
        if not mycam:
            raise RuntimeError("ONVIFカメラに接続できません")
        return mycam

    def move_initial_position(self):
        # ONVIFカメラに接続
        mycam = self._connect_onvif_camera()

        # PTZサービスを取得
        ptz = mycam.create_ptz_service()

        # プリセットポジションの一覧を取得
        presets = ptz.GetPresets({"ProfileToken": "profile1"})
        for preset in presets:
            print(f"Preset Name: {preset.Name}, Token: {preset.token}")

        # カメラをプリセットポジションに移動
        if presets:
            preset_token = presets[0].token  # 最初のプリセットを使用
            ptz.GotoPreset({"ProfileToken": "profile1", "PresetToken": preset_token})
        else:
            print("プリセットポジションが見つかりません")

    def pan_tilt(self, dir, duration=0.2):
        # ONVIFカメラに接続
        mycam = self._connect_onvif_camera()

        # 移動方向に応じたパン・チルトの速度を設定
        # (速度設定1.0以外では異音が出る。回転角はtimesecで調整すること)
        if dir == "up":
            x, y = 0.0, 1.0
        elif dir == "down":
            x, y = 0.0, -1.0
        elif dir == "left":
            x, y = -1.0, 0.0
        elif dir == "right":
            x, y = 1.0, 0.0

        # PTZサービスを取得
        ptz = mycam.create_ptz_service()

        # カメラをパン・チルト
        request = ptz.create_type("ContinuousMove")
        request.ProfileToken = "profile1"
        request.Velocity = {"PanTilt": {"x": x, "y": y}, "Zoom": {"x": 0.0}}
        ptz.ContinuousMove(request)
        time.sleep(duration)  # 移動時間
        ptz.Stop({"ProfileToken": "profile1"})
