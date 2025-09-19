import cv2
import mediapipe as mp
import math

mp_face_mesh = mp.solutions.face_mesh

# -----------------------------
# Landmark index constants (reused by new helpers)
# -----------------------------
LEFT_EYE_LOWER = {263, 374}  # {263, 249, 390, 373, 374, 380, 381, 382}
LEFT_EYE_UPPER = {362, 386}  # {362, 398, 384, 385, 386, 387, 388, 466}
RIGHT_EYE_LOWER = {33, 145}  # {33, 7, 163, 144, 145, 153, 154, 155}
RIGHT_EYE_UPPER = {133, 159}  # {133, 173, 157, 158, 159, 160, 161, 246}
NOSE = {1, 2, 4, 168, 6, 197, 195, 5}
MOUTH = {61, 291, 13, 14}
FACE_CONTOUR = (
    {10, 338, 297, 332, 284, 251, 389, 356, 454}  # 左上
    | {323, 361, 288, 397, 365, 379, 378, 400, 377}  # 左下
    | {152, 148, 176, 149, 150, 136, 172, 58, 132}  # 右下
    | {93, 234, 127, 162, 21, 54, 103, 67, 109}  # 右上
)
NOSE_TIP_IDX = 1
LEFT_EYE_BOTH = LEFT_EYE_LOWER | LEFT_EYE_UPPER
RIGHT_EYE_BOTH = RIGHT_EYE_LOWER | RIGHT_EYE_UPPER


def determine_face_orientation(landmarks, image_width, image_height):
    """
    顔の向きを推定します。
    入力:
      - landmarks: Mediapipe の face_landmarks オブジェクト (一つの顔のランドマーク集合)
      - image_width, image_height: 画像サイズ
    出力:
      {'yaw': float, 'pitch': float, 'roll': float, 'orientation': 'frontal'|'left'|'right'|'up'|'down'|'tilted'}
    """
    if landmarks is None:
        return None

    w, h = image_width, image_height

    # 左右の目の座標を取得
    left_pts = []
    for idx in LEFT_EYE_BOTH:
        lm = landmarks.landmark[idx]
        left_pts.append((lm.x * w, lm.y * h))

    right_pts = []
    for idx in RIGHT_EYE_BOTH:
        lm = landmarks.landmark[idx]
        right_pts.append((lm.x * w, lm.y * h))

    if not left_pts or not right_pts:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "orientation": "unknown"}

    # 目の中心座標を計算
    left_center_x = sum(p[0] for p in left_pts) / len(left_pts)
    left_center_y = sum(p[1] for p in left_pts) / len(left_pts)
    right_center_x = sum(p[0] for p in right_pts) / len(right_pts)
    right_center_y = sum(p[1] for p in right_pts) / len(right_pts)

    # 眉間の位置を取得
    eye_dist = math.hypot(
        left_center_x - right_center_x, left_center_y - right_center_y
    )
    center_eye_x = (left_center_x + right_center_x) / 2.0
    center_eye_y = (left_center_y + right_center_y) / 2.0

    # 鼻先の位置を取得
    nose = landmarks.landmark[NOSE_TIP_IDX]
    nose_x = nose.x * w
    nose_y = nose.y * h

    # yaw, pitch, roll を計算
    dx = nose_x - center_eye_x
    dy = nose_y - center_eye_y
    yaw = math.degrees(math.atan2(dx, eye_dist)) if eye_dist > 1e-6 else 0.0
    pitch = math.degrees(math.atan2(dy, eye_dist)) if eye_dist > 1e-6 else 0.0
    dy_eye = left_center_y - right_center_y
    roll = math.degrees(math.atan2(dy_eye, eye_dist)) if eye_dist > 1e-6 else 0.0

    # カテゴリ分類
    orientation = "frontal"
    if yaw > 20:
        orientation = "left"
    elif yaw < -20:
        orientation = "right"
    elif pitch > 35:
        orientation = "down"
    elif pitch < 10:
        orientation = "up"
    else:
        orientation = "frontal"

    return {"yaw": yaw, "pitch": pitch, "roll": roll, "orientation": orientation}


def is_eyes_closed(landmarks, image_width, image_height, threshold=0.20):
    """
    両眼が閉じているかを判定します。
    入力:
      - landmarks: Mediapipe の face_landmarks オブジェクト
      - image_width, image_height: 画像サイズ
      - threshold: EAR に相当する指標の閾値
    出力:
      {'eyes_closed': bool, 'left_eye_ear': float, 'right_eye_ear': float}
    """
    w, h = image_width, image_height
    # 左目
    ## 上まぶたの中心座標
    lm = landmarks.landmark[386]
    eye_left_upper = [lm.x * w, lm.y * h]

    ## 下まぶたの中心座標
    lm = landmarks.landmark[374]
    eye_left_lower = [lm.x * w, lm.y * h]

    ## 2点距離を計算
    left_height = math.hypot(
        eye_left_upper[0] - eye_left_lower[0], eye_left_upper[1] - eye_left_lower[1]
    )

    ## 目頭の中心座標
    lm = landmarks.landmark[133]
    eye_left_inner = [lm.x * w, lm.y * h]

    ## 目尻の中心座標
    lm = landmarks.landmark[33]
    eye_left_outer = [lm.x * w, lm.y * h]

    ## 2点距離を計算
    left_width = math.hypot(
        eye_left_inner[0] - eye_left_outer[0], eye_left_inner[1] - eye_left_outer[1]
    )

    ## アスペクト比
    left_ear = left_height / left_width if left_width > 1e-6 else 0.0

    # 右目
    ## 上まぶたの中心座標
    lm = landmarks.landmark[159]
    eye_right_upper = [lm.x * w, lm.y * h]

    ## 下まぶたの中心座標
    lm = landmarks.landmark[145]
    eye_right_lower = [lm.x * w, lm.y * h]

    ## 2点距離を計算
    right_height = math.hypot(
        eye_right_upper[0] - eye_right_lower[0], eye_right_upper[1] - eye_right_lower[1]
    )

    ## 目頭の中心座標
    lm = landmarks.landmark[362]
    eye_right_inner = [lm.x * w, lm.y * h]

    ## 目尻の中心座標
    lm = landmarks.landmark[263]
    eye_right_outer = [lm.x * w, lm.y * h]

    ## 2点距離を計算
    right_width = math.hypot(
        eye_right_inner[0] - eye_right_outer[0], eye_right_inner[1] - eye_right_outer[1]
    )

    ## アスペクト比
    right_ear = right_height / right_width if right_width > 1e-6 else 0.0

    # 閾値と比較して閉じているかを判定
    eyes_closed = (left_ear < threshold) and (right_ear < threshold)

    # 結果を辞書で返す
    return {
        "eyes_closed": eyes_closed,
        "left_eye_ear": left_ear,
        "right_eye_ear": right_ear,
    }


def is_mouth_closed(landmarks, image_width, image_height, threshold=0.20):
    """
    口が閉じているかを判定します。
    入力:
      - landmarks: Mediapipe の face_landmarks オブジェクト
      - image_width, image_height: 画像サイズ
      - threshold: mouth opening_ratio の閾値
    出力:
      {'mouth_closed': bool, 'opening_ratio': float}
    """
    w, h = image_width, image_height

    # 口の立幅
    lm = landmarks.landmark[13]
    mouth_upper = [lm.x * w, lm.y * h]
    lm = landmarks.landmark[14]
    mouth_lower = [lm.x * w, lm.y * h]
    open_dist = math.hypot(
        mouth_upper[0] - mouth_lower[0], mouth_upper[1] - mouth_lower[1]
    )

    # 口の横幅
    lm = landmarks.landmark[61]
    mouth_left = [lm.x * w, lm.y * h]
    lm = landmarks.landmark[291]
    mouth_right = [lm.x * w, lm.y * h]
    width_dist = math.hypot(
        mouth_left[0] - mouth_right[0], mouth_left[1] - mouth_right[1]
    )

    # 開口率
    opening_ratio = open_dist / width_dist if width_dist > 1e-6 else 0.0

    # 閾値と比較して閉じているかを判定
    mouth_closed = opening_ratio < threshold

    return {"mouth_closed": mouth_closed, "opening_ratio": opening_ratio}


# -----------------------------
# Existing face landmark visualization (unchanged)
# -----------------------------
def _detect_face_mesh(frame):
    """
    mediapipe face meshを使って顔の点群を検出し、(x, y, z)のリストを返す
    """
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=1, refine_landmarks=True
    )
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔検出
    results = face_mesh.process(rgb_frame)

    # close
    face_mesh.close()

    return results


def to_mesh_frame(frame):
    """
    Mediapipe face meshを使って顔のランドマークを検出し、画像へ色分けして描画する
    左目、右目、鼻、口を色分けして描画します
    """
    # 画像コピー
    frame_copy = frame.copy()
    h, w = frame.shape[:2]

    # 顔検出
    results = _detect_face_mesh(frame)

    if results.multi_face_landmarks:
        # 検出時の処理
        for face_landmarks in results.multi_face_landmarks:
            for idx, landmark in enumerate(face_landmarks.landmark):
                # 各ランドマークの色分け
                if idx in LEFT_EYE_UPPER:
                    color = (0, 255, 0)
                elif idx in LEFT_EYE_LOWER:
                    color = (0, 255, 255)
                elif idx in RIGHT_EYE_UPPER:
                    color = (0, 255, 0)
                elif idx in RIGHT_EYE_LOWER:
                    color = (0, 255, 255)
                elif idx in NOSE:
                    color = (0, 0, 255)
                elif idx in MOUTH:
                    color = (0, 255, 255)
                elif idx in FACE_CONTOUR:
                    color = (255, 255, 255)
                else:
                    continue

                # 画素座標に変換
                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # 元画像に重ねて描画
                cv2.circle(frame_copy, (x, y), 5, color, -1)
        return frame_copy

    else:
        # 検出なしは元画像をそのまま返す
        return frame
