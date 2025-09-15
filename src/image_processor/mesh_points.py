import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh


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


def detect_face_landmark(frame):
    """
    Mediapipe face meshを使って顔のランドマークを検出し、画像へ色分けして描画する
    左目、右目、鼻、口を色分けして描画します
    """
    # 画像コピー
    frame_copy = frame.copy()
    h, w = frame.shape[:2]

    # 顔検出
    results = _detect_face_mesh(frame)

    # 左目
    LEFT_EYE_LOWER = {263, 249, 390, 373, 374, 380, 381, 382}
    LEFT_EYE_UPPER = {362, 398, 384, 385, 386, 387, 388, 466}
    
    # 右目
    RIGHT_EYE_LOWER = {33, 7, 163, 144, 145, 153, 154, 155}
    RIGHT_EYE_UPPER = {133, 173, 157, 158, 159, 160, 161, 246}
    
    # 鼻、口、顔の輪郭
    NOSE = {1, 2, 4, 168, 6, 197, 195, 5}
    MOUTH = {61, 291, 13, 14}
    FACE_CONTOUR = (
        {10, 338, 297, 332, 284, 251, 389, 356, 454}  # 左上
        | {323, 361, 288, 397, 365, 379, 378, 400, 377}  # 左下
        | {152, 148, 176, 149, 150, 136, 172, 58, 132}  # 右下
        | {93, 234, 127, 162, 21, 54, 103, 67, 109}  # 右上
    )

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
                cv2.circle(frame_copy, (x, y), 3, color, -1)
        return frame_copy
    
    else:
        # 検出なしは元画像をそのまま返す
        return frame
    
