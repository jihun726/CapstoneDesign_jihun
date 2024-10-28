import torch
import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# 기본 카메라 열기 (장치 번호 2 사용)
cap = cv2.VideoCapture(2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLOv5 모델을 사용해 객체 탐지 실행
    results = model(frame)

    # 사람만 필터링하여 탐지 결과 표시
    for result in results.xyxy[0]:  # 결과는 [xmin, ymin, xmax, ymax, confidence, class]
        x1, y1, x2, y2, confidence, cls = map(int, result[:6])

        # 클래스 ID가 0일 경우만 처리 (0: 사람)
        if cls == 0:
            label = results.names[cls]  # 클래스 이름 가져오기

            # 바운딩 박스 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 화면 출력
    cv2.imshow('YOLOv5 Object Detection - Person Only', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
