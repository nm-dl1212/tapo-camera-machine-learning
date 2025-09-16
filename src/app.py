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
フレーム取得エンドポイント
"""
from src.frame import get_frame, frame_generator

@app.get("/snapshot")
def snapshot():
    frame_bytes = get_frame()
    if frame_bytes is None:
        return {"error": "フレームを取得できませんでした"}
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/video")
def video_feed():
    return StreamingResponse(
        frame_generator(detect_face_landmark),
        media_type="multipart/x-mixed-replace; boundary=frame"  
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
)


@app.get("/face")
def face():
    frame_bytes = get_frame(detect_face_landmark)
    if frame_bytes is None:
        return {"error": "フレームを取得できませんでした"}
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/face_emotion")
def face():
    frame_bytes = get_frame(detect_face_with_emotion)
    if frame_bytes is None:
        return {"error": "フレームを取得できませんでした"}
    return Response(content=frame_bytes, media_type="image/jpeg")



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
