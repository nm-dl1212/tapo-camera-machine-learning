import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse
import cv2
import time
import asyncio
from src.camera import CamPtz
from dotenv import load_dotenv

# 環境変数を.envから読み込む場合
if os.path.exists(".env"):
    load_dotenv()

app = FastAPI()

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


@app.post("/connect")
async def connect():
    try:
        await asyncio.to_thread(camera.open)  # RTSP 接続開始、スレッドでフレーム更新
        await asyncio.to_thread(camera.setup_ptz)  # PTZ 設定
        return {"status": "connected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/disconnect")
async def disconnect():
    try:
        await asyncio.to_thread(camera.close)
        return {"status": "disconnected"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
PTZ (パン・チルト・ズーム) 操作エンドポイント
"""


@app.post("/ptz")
async def ptz(request: Request):
    data = await request.json()
    if not data:
        raise HTTPException(status_code=400, detail="No JSON payload")

    x = float(data.get("x", 0))
    y = float(data.get("y", 0))

    # PTZ範囲にクランプ
    x = max(min(x, camera.XMAX), camera.XMIN)
    y = max(min(y, camera.YMAX), camera.YMIN)

    try:
        await asyncio.to_thread(camera.move, x, y)
        return {"status": "moving", "x": x, "y": y}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
単一フレーム取得エンドポイント
"""


@app.get("/frame")
async def get_frame():
    frame = await asyncio.to_thread(camera.read)
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    # JPEGエンコード
    ret, jpeg = cv2.imencode(".jpg", frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(jpeg.tobytes(), media_type="image/jpeg")


"""
ストリーミング用のエンドポイント
"""


async def gen_mjpeg():
    """最新フレームだけ返すジェネレータ"""
    while True:
        frame = await asyncio.to_thread(camera.read)
        if frame is None:
            await asyncio.sleep(0.1)
            continue

        # 最新のフレームを JPEG に変換
        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        )

        # 軽く待機（14FPSくらいに制御）
        await asyncio.sleep(0.1)


@app.get("/video")
async def video_feed():
    return StreamingResponse(
        gen_mjpeg(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


"""
以下、顔検出などの画像処理エンドポイント

"""
from src.image_processor.emotion import detect_face_with_emotion
from src.image_processor.mesh_points import (
    detect_face_landmark,
    determine_face_orientation,
    is_eyes_closed,
    is_mouth_closed,
    _detect_face_mesh,
)


@app.get("/face")
async def face():
    frame = await asyncio.to_thread(camera.read)
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    # 顔検出、表情検出
    # NOTE: 画像処理を差し込むときはここを変更する
    result_frame = await asyncio.to_thread(detect_face_with_emotion, frame)

    # JPEGエンコード
    ret, jpeg = cv2.imencode(".jpg", result_frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(jpeg.tobytes(), media_type="image/jpeg")


@app.get("/face_status")
async def face_status():
    frame = await asyncio.to_thread(camera.read)
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    results = await asyncio.to_thread(_detect_face_mesh, frame)
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        width, height = frame.shape[1], frame.shape[0]
        orientation = await asyncio.to_thread(
            determine_face_orientation, face_landmarks, width, height
        )
        eyes = await asyncio.to_thread(is_eyes_closed, face_landmarks, width, height)
        mouth = await asyncio.to_thread(is_mouth_closed, face_landmarks, width, height)
        data = {
            "orientation": orientation,
            "eyes_closed": eyes["eyes_closed"],
            "left_eye_ear": eyes["left_eye_ear"],
            "right_eye_ear": eyes["right_eye_ear"],
            "mouth_closed": mouth["mouth_closed"],
            "mouth_opening_ratio": mouth["opening_ratio"],
        }
        return data
    else:
        return {"landmarks_present": False}


@app.get("/face_points")
async def face_points():
    frame = await asyncio.to_thread(camera.read)
    if frame is None:
        raise HTTPException(status_code=503, detail="No frame available")

    result_frame = await asyncio.to_thread(detect_face_landmark, frame)

    # JPEGエンコード
    ret, jpeg = cv2.imencode(".jpg", result_frame)
    if not ret:
        raise HTTPException(status_code=500, detail="Failed to encode frame")

    return Response(jpeg.tobytes(), media_type="image/jpeg")


@app.get("/")
async def index():
    return """
    <html>
        <body>
            <h2>Tapo C210 MJPEG Viewer</h2>
            <img src="/video" width="640" />
        </body>
    </html>
    """


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
