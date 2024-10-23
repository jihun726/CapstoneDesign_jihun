import cv2
import numpy as np

# YOLO 모델 로드
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indexes = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indexes.flatten()]

# 클래스 이름 로드
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]  # 잘못된 부분 수정

# 마우스 이벤트를 처리하기 위한 변수들
points = []  # 선택된 지점들 저장
MAX_POINTS = 5  # 최대 5개의 지점을 선택할 수 있음
person_count = 0  # 사람 카운트 변수

# 마우스 콜백 함수
def select_points(event, x, y, flags, param):
    global points, person_count

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < MAX_POINTS:
            points.append((x, y))  # 지점 추가
        if len(points) == MAX_POINTS:
            points = []  # 지점이 5개가 되면 초기화
            person_count = 0  # 초기화할 때 사람 카운트도 초기화

# 비디오 캡처 시작
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# 마우스 콜백 함수 설정
cv2.namedWindow("Imagem")
cv2.setMouseCallback("Imagem", select_points)

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # YOLO를 위한 전처리
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    centers = []

    # 탐지된 객체 처리
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # 사람 클래스만 탐지
                center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                # 지정된 영역 안에 있을 경우만 탐지
                if len(points) >= 3:
                    # 클릭된 지점들을 연결하여 다각형 영역을 형성
                    pts = np.array(points, np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    is_inside = cv2.pointPolygonTest(pts, (center_x, center_y), False)
                    if is_inside < 0:  # 영역 밖이면 무시
                        continue

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                centers.append((center_x, center_y))

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    person_count = len(indexes)  # 탐지된 사람 수 업데이트
    count_displayed = 0  # 화면에 표시할 사람 수

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, (0, 255, 0), 2)

            # 각 사람의 중심 좌표와 너비, 높이를 프린트로 기록
            center_x, center_y = centers[i]
            print(f"Person {count_displayed + 1} Center: ({center_x}, {center_y}), Width: {w}, Height: {h}")
            count_displayed += 1

    # 사람 카운트 화면에 표시
    cv2.putText(frame, f"People detected: {person_count}", (10, 50), font, 2, (0, 255, 0), 2)

    # 선택된 지점들을 연결하여 다각형 그리기
    if len(points) > 1:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(255, 0, 0), thickness=2)

    cv2.imshow("Imagem", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
