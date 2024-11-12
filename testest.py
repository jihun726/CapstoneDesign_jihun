import pandas as pd
import cv2
import numpy as np
import torch
import datetime
import time
import warnings

# 경고 메시지 숨기기
warnings.filterwarnings("ignore")

# 엑셀 파일에서 데이터 불러오기
df = pd.read_excel('dot_pattern_coordinates.xlsx')
rgb_points = df[['RGB_X', 'RGB_Y']].values
thermal_points = df[['Thermal_X', 'Thermal_Y']].values

# 호모그래피 매트릭스 계산
H, _ = cv2.findHomography(rgb_points, thermal_points, method=cv2.RANSAC)
print("Homography matrix:\n", H)

# YOLO 모델 불러오기
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# 카메라 초기화
cap_rgb = cv2.VideoCapture(2)  # RGB 카메라
cap_thermal = cv2.VideoCapture(1)  # 열화상 카메라

# 안전 구역 좌표와 알림 시간 저장을 위한 딕셔너리 초기화
safety_zone = []
alert_times = {}

# 마우스 클릭 콜백 함수
def draw_safety_zone(event, x, y, flags, param):
    global safety_zone
    if event == cv2.EVENT_LBUTTONDOWN:
        # 클릭할 때마다 좌표 추가
        safety_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN and len(safety_zone) > 0:
        # 오른쪽 클릭으로 마지막 점 제거
        safety_zone.pop()

# 윈도우 이름 설정 및 크기 조절 가능하게 하기
cv2.namedWindow("Aligned and Monitored Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Aligned and Monitored Image", draw_safety_zone)

# 메인 처리 루프
while True:
    # RGB와 열화상 프레임 캡처
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_thermal, frame_thermal = cap_thermal.read()

    if not ret_rgb or not ret_thermal:
        break

    # 호모그래피를 사용하여 RGB 프레임을 열화상 프레임에 정렬
    if H is not None:
        aligned_rgb = cv2.warpPerspective(frame_rgb, H, (frame_thermal.shape[1], frame_thermal.shape[0]))
        result_image = cv2.addWeighted(aligned_rgb, 0.5, frame_thermal, 0.5, 0)

        # YOLO로 정렬된 이미지에서 객체 탐지
        results = model(result_image)
        detections = results.xyxy[0].numpy()  # 탐지된 객체의 바운딩 박스 정보

        # 탐지된 객체 처리
        for det in detections:
            x1, y1, x2, y2, conf, cls = map(int, det[:6])
            label = int(cls)
            if label == 0:  # YOLO에서 '사람' 클래스
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(result_image, f"Person: {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 사람이 안전 구역 바깥에 있는지 확인
                if len(safety_zone) >= 3:  # 안전 구역이 최소 3개의 점으로 구성된 경우
                    is_inside = cv2.pointPolygonTest(np.array(safety_zone, np.int32), (center_x, center_y), False)
                    if is_inside < 0:
                        current_time = time.time()
                        # 5초 이내에 같은 위치에서 경고를 다시 표시하지 않도록 설정
                        if (center_x, center_y) not in alert_times or current_time - alert_times[(center_x, center_y)] >= 5:
                            alert_times[(center_x, center_y)] = current_time
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Danger alert - Person at ({center_x},{center_y}) left the zone at {timestamp}")

        # 안전 구역 표시
        if len(safety_zone) >= 3:
            pts = np.array(safety_zone, np.int32).reshape((-1, 1, 2))
            cv2.polylines(result_image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.putText(result_image, "Safety Zone", (safety_zone[0][0], safety_zone[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # 결과 이미지 표시
        cv2.imshow("Aligned and Monitored Image", result_image)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap_rgb.release()
cap_thermal.release()
cv2.destroyAllWindows()
