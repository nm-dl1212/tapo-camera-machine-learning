import os
from dotenv import load_dotenv

if os.path.exists(".env"):
    load_dotenv()

# 環境変数より取得
CAMERA = os.environ["CAMERA"]
PASSWORD = os.environ["PASSWORD"]
IP_ADDRESS = os.environ["IP_ADDRESS"]
PORT = os.environ["PORT"]
STREAM = os.environ.get("STREAM")
ONVIF_PORT = os.environ["ONVIF_PORT"]

RTSP_URL = f"rtsp://{CAMERA}:{PASSWORD}@{IP_ADDRESS}:{PORT}/{STREAM}"
