import streamlit as st
import requests

import os
import json
import uuid
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

CAMERA_SERVER_URL = os.environ["CAMERA_SERVER_URL"]
CAMERA_API_URL = os.environ["CAMERA_API_URL"]
EVENT_SERVER_URL = os.environ["EVENT_SERVER_URL"]


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


"""
描画コンポーネント
"""
st.set_page_config(page_title="Camera Control", layout="wide")
st.title("🎥 Camera Control Dashboard")

# カメラ移動ボタン
st.subheader("カメラ移動コントロール")
col1, col2, col3 = st.columns([1, 2, 1])
with col1:
    if st.button("⬆️ 上へ"):
        move_camera("up")
with col2:
    left_col, right_col = st.columns([1, 1])
    with left_col:
        if st.button("⬅️ 左へ"):
            move_camera("left")
    with right_col:
        if st.button("➡️ 右へ"):
            move_camera("right")
with col3:
    if st.button("⬇️ 下へ"):
        move_camera("down")

# モード選択
mode = st.radio(
    "モードを選択してください",
    ("静止画", "ストリーミング", "動体検知"),
    horizontal=True,
)

# --- モードに応じた処理 ---
if mode == "静止画":
    snapshot_mode = st.selectbox(
        "スナップショットモードを選択してください", ("標準", "メッシュ")
    )
    if snapshot_mode == "標準":
        url = f"{CAMERA_SERVER_URL}/snapshot"
    elif snapshot_mode == "メッシュ":
        url = f"{CAMERA_SERVER_URL}/snapshot?mode=mesh"

    response = requests.get(url)
    if response.status_code == 200:
        st.image(response.content, caption="Snapshot", width="content")
    else:
        st.error("スナップショットを取得できませんでした")

elif mode == "ストリーミング":
    # ストリーミング映像表示
    html_code = f"""
        <img src="{CAMERA_API_URL}/video" height="600" />
    """
    st.components.v1.html(html_code, height=600)

elif mode == "動体検知":
    # ストリーミング映像表示
    html_code = f"""
        <img src="{CAMERA_API_URL}/video" height="600" />
    """
    st.components.v1.html(html_code, height=600)

    # 動体検知
    msg_placeholder = st.empty()
    prev_motion = False  # 直前の状態を記録

    for event in listen_to_events():
        motion = event.get("motion", False)

        if motion and not prev_motion:  # False → True に変わった瞬間だけ
            msg_placeholder.warning(f"⚠️ 動体検知！ ({event['timestamp']})")

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

        elif not motion:
            msg_placeholder.info("動きなし")

        prev_motion = motion  # 状態を更新
