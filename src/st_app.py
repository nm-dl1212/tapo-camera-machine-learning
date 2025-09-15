import streamlit as st
import requests
from PIL import Image
import io

BASE_URL = "http://localhost:5000"

st.title("📷 Tapo C210 カメラ操作 UI")

# --- 接続操作 ---
st.subheader("接続操作")
col1, col2 = st.columns(2)
with col1:
    if st.button("🔌 Connect"):
        res = requests.post(f"{BASE_URL}/connect")
        st.write(res.json())
with col2:
    if st.button("❌ Disconnect"):
        res = requests.post(f"{BASE_URL}/disconnect")
        st.write(res.json())

# --- PTZ 操作 ---
st.subheader("PTZ 操作")
x = st.slider("X 位置", -1.0, 1.0, 0.0, step=0.1)
y = st.slider("Y 位置", -1.0, 1.0, 0.0, step=0.1)

if st.button("🎛 Move Camera"):
    res = requests.post(f"{BASE_URL}/ptz", json={"x": x, "y": y})
    st.write(res.json())

# --- フレーム表示 ---
st.subheader("📸 フレーム取得")

col3, col4 = st.columns(2)

with col3:
    if st.button("📷 最新フレーム"):
        res = requests.get(f"{BASE_URL}/frame")
        if res.status_code == 200:
            img = Image.open(io.BytesIO(res.content))
            st.image(img, caption="Latest Frame")
        else:
            st.error(res.json())

with col4:
    if st.button("😊 顔検出フレーム"):
        res = requests.get(f"{BASE_URL}/face")
        if res.status_code == 200:
            img = Image.open(io.BytesIO(res.content))
            st.image(img, caption="Face Detection Frame")
        else:
            st.error(res.json())

# --- ストリーミング表示 ---
st.subheader("🎥 ライブストリーミング")

st.markdown(
    f"""
    <img src="{BASE_URL}/video" width="640" />
    """,
    unsafe_allow_html=True,
)
