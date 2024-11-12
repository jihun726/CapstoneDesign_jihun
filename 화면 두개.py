import cv2

camera = cv2.VideoCapture(1)
thermal = cv2.VideoCapture(0)

while True:
    rgb_ret, rgb_frame = camera.read()
    thermal_ret, thermal_frame = thermal.read()
    if rgb_ret:
        cv2.imshow('rgb', rgb_frame)
        cv2.imshow('thermal', thermal_frame)

        cv2.resize(rgb_frame, (320,240))

    if cv2.waitKey(10) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
        break



cv2.destroyAllWindows()
'''ss
def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"카메라 {camera_index}를 열 수 없습니다.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"카메라 {camera_index}에서 프레임을 가져올 수 없습니다.")
            break

        cv2.imshow(f"Camera {camera_index}", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 'q' 키를 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

# 카메라 인덱스 조정
camera_indexes = [1, 2]  # 두 개의 카메라 인덱스

threads = []
for index in camera_indexes:
    thread = threading.Thread(target=open_camera, args=(index,))
    thread.start()
    threads.append(thread)

# 모든 스레드가 종료될 때까지 기다림
for thread in threads:
    thread.join()
'''