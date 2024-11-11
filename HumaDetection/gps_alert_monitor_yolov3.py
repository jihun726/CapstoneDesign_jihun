import torch
import cv2
import warnings
import datetime
import numpy as np
import random  # GPS 좌표 예시용
import time

warnings.filterwarnings("ignore", category=FutureWarning)

# YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# 기본 카메라 열기 (장치 번호 2 사용)
cap = cv2.VideoCapture(0)

# 변수 초기화
safety_zone = []  # 다각형 안전 구역 포인트
person_centers = {}
alert_times = {}  # 이탈 알림 시간 저장

# 예시 GPS 위치 함수 (실제 GPS 모듈로 교체 필요)
def get_gps_location():
    lat = 34.795 + random.uniform(-0.001, 0.001)  # 예시 위도
    lon = 126.388 + random.uniform(-0.001, 0.001)  # 예시 경도
    return f"위도: {lat:.6f}, 경도: {lon:.6f}"

# 마우스 콜백 함수: 안전 구역을 다각형으로 정의하거나 초기화하는 역할
def mouse_callback(event, x, y, flags, param):
    global safety_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭할 때마다 다각형 꼭짓점 추가
        safety_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 오른쪽 클릭으로 안전 구역 초기화
        safety_zone = []

cv2.namedWindow('YOLOv5 Object Detection - Person Only')
cv2.setMouseCallback('YOLOv5 Object Detection - Person Only', mouse_callback)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLOv5 모델을 사용해 객체 탐지 실행
    results = model(frame)

    # 현재 프레임에서 탐지된 사람 수와 중심 초기화
    new_centers = {}

    # 사람만 필터링하여 탐지 결과 표시 (클래스 ID 0)
    for i, result in enumerate(results.xyxy[0]):  # 결과는 [xmin, ymin, xmax, ymax, confidence, class]
        x1, y1, x2, y2, confidence, cls = map(int, result[:6])

        if cls == 0:
            # 레이블 가져오기 및 중심점, 너비, 높이 계산
            label = results.names[cls]
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            width = x2 - x1
            height = y2 - y1

            # 중심점 기록
            person_id = f"person_{i + 1}"
            new_centers[person_id] = (center_x, center_y, width, height)

            # 바운딩 박스 및 레이블 그리기
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 중심점과 크기 정보 표시 및 기록
            info_text = f"{person_id}: Center({center_x},{center_y}) W:{width} H:{height}"
            print(info_text)  # 기록 출력
            cv2.putText(frame, info_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 안전 구역 외부에 있는지 확인
            if len(safety_zone) >= 3:  # 다각형이 최소 3개의 점으로 구성
                is_inside = cv2.pointPolygonTest(np.array(safety_zone, np.int32), (center_x, center_y), False)
                if is_inside < 0:  # 점이 다각형 밖에 있는 경우
                    # 현재 시간
                    current_time = time.time()
                    # 이전에 알람이 기록된 시간이 없거나, 5초가 경과한 경우 알림
                    if person_id not in alert_times or current_time - alert_times[person_id] >= 5:
                        alert_times[person_id] = current_time  # 현재 시간 업데이트
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        gps_info = get_gps_location()
                        print(f"Dangerous - {person_id} left the zone at {gps_info} at {timestamp}")

    # 사람 중심 업데이트
    person_centers = new_centers

    # 안전 구역 표시 (다각형으로 그리기)
    if len(safety_zone) >= 3:
        pts = np.array(safety_zone, np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "Safety Zone", (safety_zone[0][0], safety_zone[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 탐지된 사람 수 화면에 표시
    cv2.putText(frame, f"Person Count: {len(new_centers)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 결과 화면 출력
    cv2.imshow('YOLOv5 Object Detection - Person Only', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()