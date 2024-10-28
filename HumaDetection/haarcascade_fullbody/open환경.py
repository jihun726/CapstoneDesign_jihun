import cv2

# Haar 분류기 로드
body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# 웹캠으로 비디오 캡처 시작
cap = cv2.VideoCapture(1)  # 0은 기본 웹캠을 의미

while cap.isOpened():
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 프레임을 읽지 못하면 루프 종료

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)  # 신체 감지

    # 감지된 신체(사람) 수를 세기
    people_count = len(bodies)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)  # 사각형 그리기

    # 화면에 감지된 사람 수 표시
    cv2.putText(frame, f'People Count: {people_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Body Detection", frame)  # 결과 표시

    # 'q' 키가 눌리면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 비디오 캡처 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기

'''

##카메라 해상도 측정 코딩_해상도 160*122
import cv2

# 열화상 카메라 열기 (DirectShow 사용)
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 또는 cv2.CAP_MSMF 사용

if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 지원 되는 해상도 확인
width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"카메라의 기본 해상도: {width}x{height}")

while True:
    ret, frame = capture.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    cv2.imshow("original", frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()



import cv2

# 열화상 카메라 열기 (DirectShow 사용)
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 또는 cv2.CAP_MSMF 사용

if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 카메라 해상도를 160x122로 설정
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1600)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1220)

# 창을 생성하고 크기 조정 가능하도록 설정
cv2.namedWindow("original", cv2.WINDOW_NORMAL)  # 창 크기 조정 가능
cv2.resizeWindow("original", 640, 480)  # 초기 창 크기를 640x480으로 설정

while True:
    ret, frame = capture.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    # 열화상 카메라에서 데이터를 그레이스케일로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 변환된 프레임을 화면에 출력
    cv2.imshow("original", gray_frame)

    if cv2.waitKey(1) == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()


import cv2

# Haar Cascade 파일 경로 (OpenCV에서 제공)
face_cascade = cv2.CascadeClassifier('C:/Users/김지훈/Desktop/haarcascade_frontalface_default.xml')



# 카메라 열기
cap = cv2.VideoCapture(0)

while True:
    # 카메라에서 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일로 변환 (얼굴 인식 성능 향상을 위해)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 얼굴 인식
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 얼굴 주위에 네모 박스 그리기
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 프레임을 화면에 출력
    cv2.imshow('Face Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 및 창 닫기
cap.release()
cv2.destroyAllWindows()


import cv2

# Haar 분류기 로드
body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")  # 올바른 파일 경로 확인

# 웹캠으로 비디오 캡처 시작
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠을 의미

while cap.isOpened():
    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        break  # 프레임을 읽지 못하면 루프 종료

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 그레이스케일 변환
    bodies = body_classifier.detectMultiScale(gray, 1.1, 3)  # 신체 감지

    scale_w = 1.2  # 너비를 1.2배로 확장
    scale_h = 1.4  # 높이를 1.4배로 확장

    for (x, y, w, h) in bodies:
        new_w = int(w * scale_w)
        new_h = int(h * scale_h)
        new_x = x - int((new_w - w) / 2)  # 중심을 기준으로 좌우로 확장
        new_y = y - int((new_h - h) / 2)  # 중심을 기준으로 위아래로 확장

        cv2.rectangle(frame, (new_x, new_y), (new_x + new_w, new_y + new_h), (0, 255, 255), 2)

    cv2.imshow("Body Detection", frame)  # 결과 표시

    # 'q' 키가 눌리면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 비디오 캡처 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
'''