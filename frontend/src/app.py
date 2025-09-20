import streamlit as st
import requests
import time

BACKEND_URL = "http://backend:8000"  # FastAPI サーバーURL

st.set_page_config(page_title="Camera Control", layout="wide")

st.title("🎥 Camera Control Dashboard")

# ステート
if "streaming" not in st.session_state:
    st.session_state.streaming = False




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
        st.image(response.content, caption="Snapshot", use_container_width=True, width=800)
    else:
        st.error("スナップショットを取得できませんでした")

elif mode == "顔点群表示モード":
    url = f"{BACKEND_URL}/face"
    response = requests.get(url)
    if response.status_code == 200:
        st.image(response.content, caption="Face Mesh", use_container_width=True, width=800)
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
    st.subheader("📺 ストリーミング映像（1秒ごと更新）")

    # プレースホルダーを用意
    img_placeholder = st.empty()
    btn_placeholder = st.empty()

    # --- ボタン切り替え ---
    if not st.session_state.streaming:
        # 停止中 → 開始ボタンを表示
        if btn_placeholder.button("▶️ 開始", key="start_button"):
            st.session_state.streaming = True
            st.rerun()
    else:
        # ストリーミング中 → 停止ボタンを表示
        if btn_placeholder.button("🔴 停止", key="stop_button"):
            st.session_state.streaming = False
            st.rerun()

        # ストリーミングループ
        while st.session_state.streaming:
            url = f"{BACKEND_URL}/snapshot"
            response = requests.get(url)
            if response.status_code == 200:
                img_placeholder.image(response.content, caption="Streaming", use_container_width=True, width=800)
            else:
                st.error("ストリーミング映像を取得できませんでした")
                st.session_state.streaming = False
                break

            time.sleep(1)