import cv2
import threading
import os
import time
from flask import Flask, Response
from dotenv import load_dotenv

load_dotenv()
RTSP_URL = os.getenv("RTSP_URL")


# =====================
# カメラ設定
# =====================

# 最新フレーム格納用
latest_frame = None
frame_lock = threading.Lock()

# =====================
# カメラ読み込みスレッド
# =====================
def capture_thread():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("カメラに接続できません")
        return
    while True:
        ret, frame = cap.read()
        if ret:
            # スレッドセーフに保存
            with frame_lock:
                latest_frame = frame
        else:
            print("フレーム取得失敗")
            time.sleep(0.1)

# スレッド開始
t = threading.Thread(target=capture_thread, daemon=True)
t.start()


# =====================
# Flask Web サーバ
# =====================
app = Flask(__name__)

def gen_mjpeg():
    """MJPEG ストリーム生成"""
    global latest_frame
    while True:
        if latest_frame is None:
            time.sleep(0.05)
            continue
        with frame_lock:
            # JPEG に変換
            ret, jpeg = cv2.imencode('.jpg', latest_frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        # MJPEG の multipart レスポンス
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # 約30FPS


@app.route('/')
def index():
    return """
    <html>
        <body>
            <h2>Tapo C210 MJPEG Viewer</h2>
            <img src="/video" width="640" />
        </body>
    </html>
    """

    
@app.route('/video')
def video_feed():
    return Response(gen_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    """最新フレームを JPEG で返す"""
    global latest_frame
    if latest_frame is None:
        return "フレームがまだありません", 503
    
    with frame_lock:
        ret, jpeg = cv2.imencode('.jpg', latest_frame)
    if not ret:
        return "フレーム取得失敗", 500

    return Response(jpeg.tobytes(), mimetype='image/jpeg')


@app.route('/processed_snapshot')
def processed_snapshot():
    global latest_frame
    if latest_frame is None:
        return "No frame available", 503

    # スレッド安全にコピー
    with frame_lock:
        frame = latest_frame.copy()

    # TODO: ダミー処理: 2値化
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # JPEGに変換して返却
    ret, jpeg = cv2.imencode('.jpg', binary)
    if not ret:
        return "Failed to encode image", 500
    return Response(jpeg.tobytes(), mimetype='image/jpeg')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)
