import cv2

url = "rtsp://tapo-cam-name:password@ip-address:554/stream1"
cap = cv2.VideoCapture(url)

if not cap.isOpened():
    print("カメラに接続できませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレーム取得に失敗")
        break

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
