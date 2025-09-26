from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
import threading

from src.camera.move import pan_tilt
from src.camera.frame import get_frame, frame_generator, get_features
from src.image_processor.emotion import to_emotion_frame
from src.image_processor.mesh_points import (
    to_mesh_frame,
    extract_face_features,
)

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 本番は制限すべき
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

"""
PTZ (パン・チルト・ズーム) 操作エンドポイント
"""


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
        generator = frame_generator(stop_event)
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

from src.camera.frame import motion_generator_with_motion
@app.get("/video_motion")
async def video_motion(request: Request):
    stop_event = threading.Event()

    async def video_stream():
        generator = motion_generator_with_motion(stop_event)
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


@app.get("/face")
def face():
    frame_bytes = get_frame(transform_func=to_mesh_frame)
    if frame_bytes is None:
        return {"error": "フレームを取得できませんでした"}
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/emotion")
def emotion():
    frame_bytes = get_frame(transform_func=to_emotion_frame)
    if frame_bytes is None:
        return {"error": "フレームを取得できませんでした"}
    return Response(content=frame_bytes, media_type="image/jpeg")


@app.get("/features")
def features():
    features = get_features(extract_func=extract_face_features)
    if features is None:
        return {"error": "特徴を検知できませんでした"}
    return features


from src.camera.frame import is_motion, last_motion_time

@app.get("/event")
async def event():
    """
    現在のis_motionフラグと最後の検知時間を返すエンドポイント
    """
    last_motion_time_str = None
    if last_motion_time is not None:
        import datetime
        last_motion_time_str = datetime.datetime.fromtimestamp(last_motion_time).isoformat()

    return {
        "is_motion": is_motion,
        "last_motion_time": last_motion_time_str,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000, reload=True)
