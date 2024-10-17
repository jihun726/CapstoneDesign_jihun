import cv2
import numpy as np

# RGB 및 열화상 카메라의 인덱스 설정 (인덱스는 시스템에 따라 다를 수 있음)
rgb_camera_index = 0  # RGB 카메라
thermal_camera_index = 1  # 열화상 카메라

# 카메라 캡처 객체 생성
rgb_cap = cv2.VideoCapture(rgb_camera_index)
thermal_cap = cv2.VideoCapture(thermal_camera_index)

# 카메라가 정상적으로 열렸는지 확인
if not rgb_cap.isOpened() or not thermal_cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 보정된 카메라 매트릭스 및 왜곡 계수 설정 (이 값들은 캘리브레이션 과정에서 계산된 값이어야 함)
camera_matrix_rgb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 실제 값으로 교체해야 함
dist_coeffs_rgb = np.zeros((4, 1))  # 실제 왜곡 계수로 교체

camera_matrix_thermal = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])  # 실제 값으로 교체해야 함
dist_coeffs_thermal = np.zeros((4, 1))  # 실제 왜곡 계수로 교체

# 캘리브레이션 결과로부터 얻은 회전 및 이동 행렬
R = np.eye(3)  # 실제 캘리브레이션 결과로부터 얻어야 함
T = np.zeros((3, 1))  # 실제 캘리브레이션 결과로부터 얻어야 함

# 리매핑을 위한 매트릭스 계산
h_rgb, w_rgb = int(rgb_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(rgb_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_thermal, w_thermal = int(thermal_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(thermal_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# 리매핑을 위한 보정 계산 (새로운 카메라 매트릭스 얻기)
rectify_scale = 1  # 0은 잘린 이미지, 1은 전체 이미지
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(camera_matrix_rgb, dist_coeffs_rgb, camera_matrix_thermal, dist_coeffs_thermal, (w_rgb, h_rgb), R, T, rectify_scale)

# 왜곡 제거용 리매핑 생성
map1_rgb, map2_rgb = cv2.initUndistortRectifyMap(camera_matrix_rgb, dist_coeffs_rgb, R1, P1, (w_rgb, h_rgb), cv2.CV_16SC2)
map1_thermal, map2_thermal = cv2.initUndistortRectifyMap(camera_matrix_thermal, dist_coeffs_thermal, R2, P2, (w_thermal, h_thermal), cv2.CV_16SC2)

# 실시간 비디오 스트림 처리
while True:
    # RGB 카메라에서 프레임 읽기
    ret_rgb, frame_rgb = rgb_cap.read()
    if not ret_rgb:
        print("RGB 프레임을 읽을 수 없습니다.")
        break

    # 열화상 카메라에서 프레임 읽기
    ret_thermal, frame_thermal = thermal_cap.read()
    if not ret_thermal:
        print("열화상 프레임을 읽을 수 없습니다.")
        break

    # RGB 프레임 왜곡 보정 및 리매핑
    undistorted_rgb = cv2.remap(frame_rgb, map1_rgb, map2_rgb, cv2.INTER_LINEAR)

    # 열화상 프레임 왜곡 보정 및 리매핑
    undistorted_thermal = cv2.remap(frame_thermal, map1_thermal, map2_thermal, cv2.INTER_LINEAR)

    # 두 이미지 사이에 간단한 오버레이 처리 (이 부분을 맞춤 조정 가능)
    # 여기서는 두 영상을 나란히 배치하여 출력
    combined_frame = np.hstack((undistorted_rgb, undistorted_thermal))

    # 결과 영상 출력
    cv2.imshow('RGB and Thermal Combined', combined_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 해제 및 창 닫기
rgb_cap.release()
thermal_cap.release()
cv2.destroyAllWindows()
