
##카메라 실행 코딩
import cv2

# DirectShow를 사용하여 0번 카메라(외부 웹캠)을 열어봄
capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # 또는 cv2.CAP_MSMF 사용

if not capture.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1000) #해상도 조절
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

# 창을 생성하고 크기 조정 가능하도록 설정
cv2.namedWindow("original", cv2.WINDOW_NORMAL)  # 창 크기 조정 가능
cv2.resizeWindow("original", 500, 300)  # 초기 창 크기를 160x122으로 설정


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
'''

'''
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
'''