import cv2

"""
表情認識
"""

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")


def detect_face_with_emotion(frame):
    """
    顔検出＋表情判定＋ラベル描画
    """
    result_frame = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return result_frame  # 顔なしの場合は元画像を返す

    for x, y, w, h in faces:
        roi_gray = gray[y : y + h, x : x + w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)

        # 表情判定
        label = "Smile" if len(smiles) > 0 else "Neutral"
        score = 1.0 if label == "Smile" else 0.5

        # バウンディングボックス
        cv2.rectangle(result_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # ラベル描画
        text = f"{label} ({score:.2f})"
        y_text = y - 10 if y - 10 > 10 else y + 10
        cv2.putText(
            result_frame,
            text,
            (x, y_text),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

    return result_frame
