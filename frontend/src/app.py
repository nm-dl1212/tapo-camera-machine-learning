import streamlit as st
import requests
import time

import os
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

# 環境変数より取得
BACKEND_URL = os.environ["BACKEND_URL"]  # FastAPI サーバーURL

st.set_page_config(page_title="Camera Control", layout="wide")

st.title("🎥 Camera Control Dashboard")

# ステート
if "streaming" not in st.session_state:
    st.session_state.streaming = False

if "move_status" not in st.session_state:
    st.session_state.move_status = ""

# --- カメラ移動 ---
def move_camera(direction: str):
    try:
        response = requests.post(f"{BACKEND_URL}/pan_tilt", json={"direction": direction, "duration": 0.5})
        if response.status_code == 200:
            st.session_state.move_status = response.json().get("message", "移動成功")
        else:
            st.session_state.move_status = f"移動失敗: {response.text}"
    except Exception as e:
        st.session_state.move_status = f"エラー: {str(e)}"




# --- モード選択 ---
mode = st.radio(
    "モードを選択してください",
    ("顔点群表示モード", "静止画モード", "ストリーミングモード"),
    horizontal=True,
)

# --- モードに応じた処理 ---
if mode == "静止画モード":
    url = f"{BACKEND_URL}/snapshot"
    response = requests.get(url)
    if response.status_code == 200:
        st.image(response.content, caption="Snapshot", width="content")
    else:
        st.error("スナップショットを取得できませんでした")

elif mode == "顔点群表示モード":
    url = f"{BACKEND_URL}/face"
    response = requests.get(url)
    if response.status_code == 200:
        st.image(response.content, caption="Face Mesh", width="content")
    else:
        st.error("顔メッシュを取得できませんでした")

    # 特徴検出を追加
    feat_url = f"{BACKEND_URL}/features"
    feat_response = requests.get(feat_url)
    if feat_response.status_code == 200:
        st.subheader("特徴検出結果")
        st.json(feat_response.json())
    else:
        st.warning("特徴を検知できませんでした")

elif mode == "ストリーミングモード":
    st.subheader("📺 ストリーミング映像")
    
    # TODO: 修正検討
    # ここはブラウザからアクセスすることになるので、バックエンドの公開URLにする必要あり。
    html_code = f"""
        <img src="{BACKEND_URL}/video" height="600" />
    """
    st.components.v1.html(html_code, height=600)


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

    if st.session_state.move_status:
        st.info(st.session_state.move_status)