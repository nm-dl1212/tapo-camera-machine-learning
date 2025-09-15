
# 学習済モデルのダウンロード
物体検出(MobileNet)を使用する場合のみ
```bash
mkdir pretrain_models/
wget https://sourceforge.net/projects/ip-cameras-for-vlc/files/MobileNetSSD_deploy.prototxt/download -O pretrain_models/MobileNetSSD_deploy.prototxt
wget https://sourceforge.net/projects/ip-cameras-for-vlc/files/MobileNetSSD_deploy.caffemodel/download -O pretrain_models/MobileNetSSD_deploy.caffemodel
```

# 起動
```bash
uv run src/app.py
```