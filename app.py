import os
from flask import Flask, Response, request, jsonify
import cv2
import time

from camera import CamPtz 
from image_processor import detect_face

# 環境変数を.envから読み込む場合
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv()

app = Flask(__name__)

# 単一カメラインスタンス
camera = CamPtz(
    user=os.environ.get("CAMERA"),
    pwd=os.environ.get("PASSWORD"),
    ipaddr=os.environ.get("IP_ADDRESS"),
    port=int(os.environ.get("PORT")),
    stream=os.environ.get("STREAM"),
    onvif_port=int(os.environ.get("ONVIF_PORT")),
)

"""
カメラ接続用のエンドポイント
"""
@app.route("/connect", methods=["POST"])
def connect():
    try:
        camera.open()       # RTSP 接続開始、スレッドでフレーム更新
        camera.setup_ptz()  # PTZ 設定
        return jsonify({"status": "connected"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/disconnect", methods=["POST"])
def disconnect():
    try:
        camera.close()
        return jsonify({"status": "disconnected"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


"""
PTZ (パン・チルト・ズーム) 操作エンドポイント
"""
@app.route("/ptz", methods=["POST"])
def ptz():
    data = request.json
    if not data:
        return jsonify({"error": "No JSON payload"}), 400

    x = float(data.get("x", 0))
    y = float(data.get("y", 0))

    # PTZ範囲にクランプ
    x = max(min(x, camera.XMAX), camera.XMIN)
    y = max(min(y, camera.YMAX), camera.YMIN)

    try:
        camera.move(x, y)
        return jsonify({"status": "moving", "x": x, "y": y})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


"""
単一フレーム取得エンドポイント
"""
@app.route("/frame", methods=["GET"])
def get_frame():
    frame = camera.read()
    if frame is None:
        return jsonify({"error": "No frame available"}), 503

    # JPEGエンコード
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({"error": "Failed to encode frame"}), 500

    return Response(jpeg.tobytes(), mimetype='image/jpeg')


"""
ストリーミング用のエンドポイント
"""
def gen_mjpeg():
    """MJPEG ストリーム生成"""
    while True:
        frame = camera.read()
        if frame is None:
            time.sleep(0.05)
            continue

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.03)  # 約30FPS


@app.route("/video")
def video_feed():
    return Response(
        gen_mjpeg(),
        mimetype='multipart/x-mixed-replace; boundary=frame',
    )


@app.route('/face')
def face():
    frame = camera.read()
    if frame is None:
        return jsonify({"error": "No frame available"}), 503

    # 顔検出、表情検出
    # NOTE: 画像処理を差し込むときはここを変更する
    result_frame = detect_face(frame)

    # JPEGエンコード
    ret, jpeg = cv2.imencode('.jpg', result_frame)
    if not ret:
        return jsonify({"error": "Failed to encode frame"}), 500

    return Response(jpeg.tobytes(), mimetype='image/jpeg')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
