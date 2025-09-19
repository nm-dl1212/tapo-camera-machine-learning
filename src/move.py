import os
import time
from onvif import ONVIFCamera

# 環境変数
CAMERA = os.environ["CAMERA"]
PASSWORD = os.environ["PASSWORD"]
IP_ADDRESS = os.environ["IP_ADDRESS"]
ONVIF_PORT = os.environ["ONVIF_PORT"]


def move_initial_position():
    # ONVIFカメラに接続
    mycam = ONVIFCamera(IP_ADDRESS, ONVIF_PORT, CAMERA, PASSWORD)
    if not mycam:
        raise RuntimeError("ONVIFカメラに接続できません")

    # PTZサービスを取得
    ptz = mycam.create_ptz_service()

    # プリセットポジションの一覧を取得
    presets = ptz.GetPresets({"ProfileToken": "profile1"})
    for preset in presets:
        print(f"Preset Name: {preset.Name}, Token: {preset.token}")

    # カメラをプリセットポジションに移動
    if presets:
        preset_token = presets[0].token  # 最初のプリセットを使用
        ptz.GotoPreset({"ProfileToken": "profile1", "PresetToken": preset_token})
    else:
        print("プリセットポジションが見つかりません")


def pan_tilt(dir, duration=0.2):
    # 移動方向に応じたパン・チルトの速度を設定
    # (速度設定1.0以外では異音が出る。回転角はtimesecで調整すること)
    if dir == "up":
        x, y = 0.0, 1.0
    elif dir == "down":
        x, y = 0.0, -1.0
    elif dir == "left":
        x, y = -1.0, 0.0
    elif dir == "right":
        x, y = 1.0, 0.0

    # ONVIFカメラに接続
    mycam = ONVIFCamera(IP_ADDRESS, ONVIF_PORT, CAMERA, PASSWORD)
    if not mycam:
        raise RuntimeError("ONVIFカメラに接続できません")

    # PTZサービスを取得
    ptz = mycam.create_ptz_service()

    # カメラをパン・チルト
    request = ptz.create_type("ContinuousMove")
    request.ProfileToken = "profile1"
    request.Velocity = {"PanTilt": {"x": x, "y": y}, "Zoom": {"x": 0.0}}
    ptz.ContinuousMove(request)
    time.sleep(duration)  # 移動時間
    ptz.Stop({"ProfileToken": "profile1"})
