
# モデルのダウンロード
wget https://sourceforge.net/projects/ip-cameras-for-vlc/files/MobileNetSSD_deploy.prototxt/download -O MobileNetSSD_deploy.prototxt
wget https://sourceforge.net/projects/ip-cameras-for-vlc/files/MobileNetSSD_deploy.caffemodel/download -O MobileNetSSD_deploy.caffemodel


# 起動
uv run app.py