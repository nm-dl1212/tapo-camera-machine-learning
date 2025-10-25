import streamlit as st
import requests

import os
import json
import uuid
import time
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

CAMERA_SERVER_URL = os.environ["CAMERA_SERVER_URL"]
CAMERA_API_URL = os.environ["CAMERA_API_URL"]
EVENT_SERVER_URL = os.environ["EVENT_SERVER_URL"]

# stateの初期化
if "move_status" not in st.session_state:
    st.session_state.move_status = ""
if "detected_imgs" not in st.session_state:
    st.session_state.detected_imgs = []
if "detected_times" not in st.session_state:
    st.session_state.detected_times = []
if "prev_motion" not in st.session_state:
    st.session_state.prev_motion = False


def move_camera(direction: str):
    """
    カメラを移動させる処理
    """
    try:
        response = requests.post(
            f"{CAMERA_SERVER_URL}/pan_tilt",
            json={"direction": direction, "duration": 0.5},
        )
    except Exception as e:
        st.session_state.move_status = f"エラー: {str(e)}"


def listen_to_events():
    with requests.get(f"{EVENT_SERVER_URL}/event", stream=True) as r:
        for line in r.iter_lines():
            if line and line.startswith(b"data:"):
                data = json.loads(line.decode("utf-8").replace("data: ", ""))
                yield data

# --- Streamlit UI ---
st.set_page_config(page_title="Camera Control", layout="wide")
st.title("🎥 Camera Dashboard")

# カメラ移動ボタン
with st.popover("Move Camera"):
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1], gap="small")
    with col1:
        if st.button("👈"):
            move_camera("left")
    with col2:
        if st.button("👇"):
            move_camera("down")
    with col3:
        if st.button("👆"):
            move_camera("up")
    with col4:
        if st.button("👉"):
            move_camera("right")

# ストリーミング映像表示
html_code = f"""
    <img src="{CAMERA_API_URL}/video" style="width: 100%; height: auto;" />
"""
st.html(html_code)

# --- モードに応じた処理 ---
mode = st.radio(
    "モードを選択してください",
    ("ストリーミング", "動体検知"),
    horizontal=True,
)

if mode == "ストリーミング":
    pass

elif mode == "動体検知":
    # 動体検知
    event_placeholder = st.empty()
    motion_placeholder = st.empty()
    face_placeholder = st.empty()
    img_placeholder = st.empty()
    
    st.session_state.prev_motion = False  # 直前の状態を記録

    for event in listen_to_events():
        # state 表示
        event_placeholder.json(event, expanded=False)

        motion = event.get("motion", False)
        
        if motion and not st.session_state.prev_motion:  # False → True に変わった瞬間だけ
            motion_placeholder.warning(f"⚠️ 動体検知！ ({event['timestamp']})")

            # ブザー音を鳴らす
            div_id = str(uuid.uuid4())
            sound_html = f"""
            <div id="{div_id}"></div>
            <script>
                var audio = new Audio("https://actions.google.com/sounds/v1/cartoon/cartoon_boing.ogg");
                audio.play();
            </script>
            """
            st.components.v1.html(sound_html, height=0)

            # スナップショットとタイムスタンプを表示
            url = f"{CAMERA_SERVER_URL}/snapshot?mode=mesh"
            response = requests.get(url)

            if response.status_code == 200:
                # detected_imgs に画像を追加. 10件を超える場合は古いものを削除
                st.session_state.detected_imgs.append({"content": response.content, "timestamp": event["timestamp"]})
                if len(st.session_state.detected_imgs) > 10:
                    st.session_state.detected_imgs.pop(0)
            else:
                st.error("スナップショットを取得できませんでした")

            # detected_imgs に溜まった画像を、逆順5個まで表示
            display_imgs = st.session_state.detected_imgs[-10:][::-1]
            display_times = st.session_state.detected_times[-10:][::-1]

            with img_placeholder.expander("検知画像一覧", expanded=False):
                for i, img in enumerate(display_imgs):
                    st.image(img["content"], caption=f"検知時刻: {img["timestamp"]}", width=500)


        elif not motion:
            motion_placeholder.info("動きなし")

        st.session_state.prev_motion = motion  # 状態を更新


        # 顔検出
        face = event.get("face_detected", False)

        if face:
            eyes_status = "寝ています💤" if event["eyes_closed"] else "目覚めました🐔"
            orientation = event.get("orientation")
            orientation_ja = {"frontal": "正面", "right": "右", "left": "左", "up" : "上", "down": "下"}
            orientation = orientation_ja.get(orientation)
            face_placeholder.success(f"ぺそちが{eyes_status}  {orientation}を向いているようです👀")

        else:
            face_placeholder.error("顔検出できません⚡")

