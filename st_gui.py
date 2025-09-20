import streamlit as st
import requests
import time

BACKEND_URL = "http://localhost:8000"  # FastAPI サーバーURL

st.set_page_config(page_title="Camera Control", layout="wide")

st.title("🎥 Camera Control Dashboard")

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
        st.image(response.content, caption="Snapshot", use_container_width=True)
    else:
        st.error("スナップショットを取得できませんでした")

elif mode == "顔点群表示モード":
    url = f"{BACKEND_URL}/face"
    response = requests.get(url)
    if response.status_code == 200:
        st.image(response.content, caption="Face Mesh", use_container_width=True)
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

    # 画像を表示するプレースホルダ
    img_placeholder = st.empty()

    # Stop ボタン
    stop = st.button("⏹️ 停止", key="stop_button")

    # ループ
    while not stop:
        url = f"{BACKEND_URL}/snapshot"
        response = requests.get(url)
        if response.status_code == 200:
            img_placeholder.image(response.content, caption="Streaming", use_container_width=True)
        else:
            st.error("ストリーミング映像を取得できませんでした")
            break

        time.sleep(5)  # 1秒ごとに更新
