import cv2
import numpy as np
import glob
import torch

# 1. YOLOv5 모델 로드
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # YOLOv5 모델 로드

# 2. 체크보드 캘리브레이션 (패턴 사이즈 및 실제 좌표 설정)
chessboard_size = (9, 6)
obj_points = []  # 3D 공간의 실제 점
image_points_rgb = []  # RGB 카메라의 2D 이미지 점
image_points_thermal = []  # 열화상 카메라의 2D 이미지 점

# 실제 세계 좌표 생성
obj_point = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
obj_point[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# RGB 및 열화상 캘리브레이션 이미지 불러오기
rgb_images = glob.glob('path/to/rgb_calibration_images/*.jpg')
thermal_images = glob.glob('path/to/thermal_calibration_images/*.jpg')

for img_rgb, img_thermal in zip(rgb_images, thermal_images):
    image_rgb = cv2.imread(img_rgb)
    image_thermal = cv2.imread(img_thermal, cv2.IMREAD_GRAYSCALE)

    # 체크보드 코너 찾기
    ret_rgb, corners_rgb = cv2.findChessboardCorners(image_rgb, chessboard_size, None)
    ret_thermal, corners_thermal = cv2.findChessboardCorners(image_thermal, chessboard_size, None)

    if ret_rgb and ret_thermal:
        obj_points.append(obj_point)
        image_points_rgb.append(corners_rgb)
        image_points_thermal.append(corners_thermal)

# RGB 및 열화상 카메라의 캘리브레이션 수행
ret_rgb, camera_matrix_rgb, dist_coeff_rgb, _, _ = cv2.calibrateCamera(obj_points, image_points_rgb,
                                                                       image_rgb.shape[:2], None, None)
ret_thermal, camera_matrix_thermal, dist_coeff_thermal, _, _ = cv2.calibrateCamera(obj_points, image_points_thermal,
                                                                                   image_thermal.shape[:2], None, None)

# 3. 실시간 카메라 피드 열기
cap_rgb = cv2.VideoCapture(0)  # RGB 카메라
cap_thermal = cv2.VideoCapture(1)  # 열화상 카메라

# 카메라가 열리지 않은 경우 오류 처리
if not cap_rgb.isOpened() or not cap_thermal.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

while cap_rgb.isOpened() and cap_thermal.isOpened():
    # 각 카메라로부터 프레임 읽기
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_thermal, frame_thermal = cap_thermal.read()

    if not ret_rgb or not ret_thermal:
        print("Error: Failed to grab frame from one or both cameras.")
        break

    # 열화상 프레임을 RGB 해상도로 업스케일링 (1920x1080)
    upscaled_thermal = cv2.resize(frame_thermal, (1920, 1080), interpolation=cv2.INTER_CUBIC)

    # YOLOv5로 객체 감지 수행
    results_rgb = model(frame_rgb)
    results_thermal = model(upscaled_thermal)

    # RGB 프레임에서 객체 감지 결과 표시
    for _, row in results_rgb.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 열화상 프레임에서 객체 감지 결과 표시
    for _, row in results_thermal.pandas().xyxy[0].iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        label = f"{row['name']} {row['confidence']:.2f}"
        cv2.rectangle(upscaled_thermal, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(upscaled_thermal, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # RGB와 열화상 프레임을 가중치를 적용하여 병합 (50%씩 오버레이)
    merged_frame = cv2.addWeighted(frame_rgb, 0.6, upscaled_thermal, 0.4, 0)

    # 화면에 송출
    cv2.imshow("Merged Output (RGB + Thermal)", merged_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap_rgb.release()
cap_thermal.release()
cv2.destroyAllWindows()
