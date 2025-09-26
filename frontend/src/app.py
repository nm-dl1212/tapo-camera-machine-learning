import streamlit as st
import requests
import time
import datetime

import os
import cv2
import numpy as np
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

# 環境変数より取得
BACKEND_URL = os.environ["BACKEND_URL"]  # FastAPI サーバーURL
INTERVAL = 3  # 秒
THRESHOLD = 50  # 動体検知の閾値
MIN_AREA = 5000  # 動体検知の最小エリア


st.set_page_config(page_title="Camera Control", layout="wide")

st.title("🎥 Camera Control Dashboard")

# ステート
if "streaming" not in st.session_state:
    st.session_state.streaming = False

if "move_status" not in st.session_state:
    st.session_state.move_status = ""


if "prev_gray" not in st.session_state:
    st.session_state.prev_gray = None


if "is_detect" not in st.session_state:
    st.session_state.is_detect = False

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
    ("顔点群表示モード", "静止画モード", "ストリーミングモード", "動体検知モード"),
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


elif mode == "動体検知モード":
     # ストリーミング映像表示（動画）
    html_code = f"""
        <img src="{BACKEND_URL}/video_motion" height="400" />
    """
    st.components.v1.html(html_code, height=400)

    
    img_placeholder = st.empty()  # 画像表示用プレースホルダ
    msg_placeholder = st.empty()  # メッセージ表示用
    
    while True:
        try:
            # スナップショット取得
            resp = requests.get(f"{BACKEND_URL}/snapshot")
            img_array = np.frombuffer(resp.content, np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                continue
                
            # フレーム画像をグレースケール化＆ぼかし
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (21, 21), 0)
                
            is_detect = False
            if st.session_state.prev_gray is not None:
                # 差分計算
                frame_delta = cv2.absdiff(st.session_state.prev_gray, curr_gray)
                _, thresh_img = cv2.threshold(frame_delta, THRESHOLD, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for cnt in contours:
                    if cv2.contourArea(cnt) > MIN_AREA: # 閾値を超えたら処理を続行
                        # 動きのあった箇所を矩形で囲む
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            
                        # 動き検知フラグを立てる
                        is_detect = True

            st.session_state.prev_gray = curr_gray  # 更新
            st.session_state.detect = is_detect

            
            # メッセージ表示
            if is_detect:
                dt_now = datetime.datetime.now() + datetime.timedelta(hours=9)
                img_placeholder.image(frame, channels="BGR")
                msg_placeholder.success(f"{dt_now}: 動きが検知されました！")

            time.sleep(INTERVAL)

        except Exception as e:
            msg_placeholder.error(f"エラー: {e}")
            time.sleep(INTERVAL)