import cv2
import numpy as np

# 열화상 카메라로 변경 (카메라 1번으로 지정)
cap = cv2.VideoCapture(1)

# Safety 영역을 저장할 리스트
safety_zone = []
drawing = False

# 사람 중심 좌표 기록
person_center_records = {}

# 마우스 콜백 함수 정의 (마우스 클릭으로 안전 영역 생성)
def draw_safety_zone(event, x, y, flags, param):
    global safety_zone, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        safety_zone.append((x, y))  # 좌표 추가
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        safety_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:  # 오른쪽 클릭 시 영역 초기화
        safety_zone.clear()  # 안전 구역 초기화
        
# 창 생성 및 마우스 콜백 함수 설정
cv2.namedWindow("Thermal Camera Human Detection", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Thermal Camera Human Detection", draw_safety_zone)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame.")
        break

    # 그레이스케일 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 이진화 (Thresholding)
    _, threshold = cv2.threshold(gray_frame, 200, 255, cv2.THRESH_BINARY)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 사람 수 카운팅 초기화
    people_count = 0
    exit_detection = False  # 사람의 안전 영역 이탈 여부

    # 윤곽선을 따라 사람으로 추정되는 객체에 바운딩 박스 그리기
    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) > 500:  # 작은 노이즈 무시
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2  # 중심 좌표 X
            cy = y + h // 2  # 중심 좌표 Y

            # 사람 수 증가
            people_count += 1

            # 사람의 중심 좌표 기록 (구분 가능하게 번호 붙이기)
            person_center_records[f"Person {people_count}"] = (cx, cy)

            # 사람 주변에 네모 박스(바운딩 박스) 표시
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # 녹색 네모 박스 표시

            # 사람이 safety 영역에 있는지 확인
            if safety_zone:
                # 다각형 영역 안에 사람이 있는지 체크
                if cv2.pointPolygonTest(np.array(safety_zone, np.int32), (cx, cy), False) < 0:
                    # 영역을 벗어나면 위험(dangerous)으로 설정하고 Exit detection
                    exit_detection = True

    # 안전 영역 그리기 (선으로 표시)
    if len(safety_zone) > 1:
        cv2.polylines(frame, [np.array(safety_zone, np.int32)], isClosed=True, color=(225, 0, 0), thickness=2)

    # 사람이 영역을 벗어나면 알림
    if exit_detection:
        cv2.putText(frame, 'Dangerous', (100, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        print("Person exited the safety zone!")

    # 사람 수 표시
    cv2.putText(frame, f'People Count: {people_count}', (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # 결과 화면 출력
    cv2.imshow("Thermal Camera Human Detection", frame)

    # 실시간으로 사람 중심 좌표 기록을 콘솔에 출력 (영어로)
    for person, center in person_center_records.items():
        print(f"{person}: Center coordinates {center}")

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
