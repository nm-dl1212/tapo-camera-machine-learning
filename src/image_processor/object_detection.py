import numpy as np
import cv2

"""
ダミーの処理: 2値化した画像を返却する
"""


def binarize_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return binary


"""
MobileNet-SSD を用いた物体検出
"""
# クラスラベル
CLASSES = [
    "background",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

# カラーマップ
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# モデルの読み込み
net = cv2.dnn.readNetFromCaffe(
    "pretrain_models/MobileNetSSD_deploy.prototxt",
    "pretrain_models/MobileNetSSD_deploy.caffemodel",
)


def detect_objects(frame, confidence_threshold=0.5):
    # 入力画像のサイズ調整
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5
    )

    net.setInput(blob)
    detections = net.forward()

    # 検出結果を描画
    for i in range(detections.shape[2]):
        # 信頼度でフィルター
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            idx = int(detections[0, 0, i, 1])

            # バウンディングボックス
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)

            # ラベル
            label = f"{CLASSES[idx]}: {confidence:.2f}"
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(
                frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 3.0, COLORS[idx], 2
            )

    return frame


