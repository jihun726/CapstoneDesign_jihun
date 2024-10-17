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

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)  # 사각형 그리기

    cv2.imshow("Body Detection", frame)  # 결과 표시

    # 'q' 키가 눌리면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # 비디오 캡처 해제
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기
