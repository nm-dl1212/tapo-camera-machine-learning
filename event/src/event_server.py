from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import asyncio
import httpx
import json
import time
import cv2
import numpy as np
import os
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

CAMERA_SERVER_URL = os.environ["CAMERA_SERVER_URL"]

app = FastAPI()

# グローバルで動体検知の状態を保持
motion_state = {"motion": False, "timestamp": None}


def detect_motion(prev_frame, curr_frame, threshold=50, min_area=5000):
    """
    前フレームと現在フレームを比較して動体検知
    """
    # 前処理 (白黒化&ぼかし)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.GaussianBlur(prev_gray, (21, 21), 0)
    curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)

    # 差分抽出
    frame_delta = cv2.absdiff(prev_gray, curr_gray)
    _, thresh_img = cv2.threshold(frame_delta, threshold, 255, cv2.THRESH_BINARY)

    # 輪郭抽出
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # 十分に大きな動体があるか判定
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            return True
    return False


@app.on_event("startup")
async def startup_event():
    """
    アプリケーション起動時に動体検知ジョブを開始
    """

    # 動体検知ジョブの定義
    # 5秒おきにカメラサーバーからフレームを取得し、動体検知を行う
    async def motion_detection_job():
        global motion_state
        prev_frame = None

        async with httpx.AsyncClient() as client:
            while True:
                try:
                    resp = await client.get(f"{CAMERA_SERVER_URL}/snapshot", timeout=10)
                    arr = np.frombuffer(resp.content, dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                    if prev_frame is not None:
                        if detect_motion(prev_frame, frame):
                            motion_state = {
                                "motion": True,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            }
                        else:
                            motion_state = {"motion": False, "timestamp": None}

                    prev_frame = frame
                except Exception as e:
                    motion_state = {"motion": False, "error": str(e)}

                await asyncio.sleep(5)

    # バックグラウンドで動体検知ジョブを開始
    asyncio.create_task(motion_detection_job())


async def event_generator():
    while True:
        event = json.dumps(motion_state, ensure_ascii=False)
        yield f"data: {event}\n\n"
        await asyncio.sleep(1)


@app.get("/event")
async def event_stream():
    return StreamingResponse(event_generator(), media_type="text/event-stream")
