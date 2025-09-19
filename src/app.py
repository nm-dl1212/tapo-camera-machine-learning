import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import threading

app = FastAPI()


"""
PTZ (パン・チルト・ズーム) 操作エンドポイント
"""
from src.camera.move import pan_tilt


class PanTiltRequest(BaseModel):
    direction: str = "up"  # 移動方向 ('up', 'down', 'left', 'right')
    duration: float = 0.5  # 移動時間（秒）


@app.post("/pan_tilt")
async def ptz(request: PanTiltRequest):
    direction = request.direction
    duration = request.duration

    if direction not in ["up", "down", "left", "right"]:
        raise HTTPException(status_code=400, detail="Invalid direction")

    try:
        pan_tilt(direction, duration)
        return {
            "status": "success",
            "message": f"Camera moved {direction} for {duration} seconds",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


"""
フレーム取得エンドポイント
"""
from src.camera.frame import get_frame, frame_generator


@app.get("/snapshot")
def snapshot():
    frame_bytes = get_frame()
    if frame_bytes is None:
        return {"error": "フレームを取得できませんでした"}
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/video")
async def video_feed(request: Request):
    stop_event = threading.Event()

    # クライアントが切断した場合にストリーミングを停止するための非同期ジェネレーター
    async def video_stream():
        generator = frame_generator(stop_event, transfrom_func=detect_face_landmark)
        try:
            for chunk in generator:
                if await request.is_disconnected():
                    stop_event.set()
                    break
                yield chunk
        finally:
            stop_event.set()

    return StreamingResponse(
        video_stream(), media_type="multipart/x-mixed-replace; boundary=frame"
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
