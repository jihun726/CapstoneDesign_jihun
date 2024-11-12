import pandas as pd

# 엑셀 파일에서 데이터 읽기
df = pd.read_excel('dot_pattern_coordinates.xlsx')

# RGB와 Thermal 좌표 추출
rgb_points = df[['RGB_X', 'RGB_Y']].values
thermal_points = df[['Thermal_X', 'Thermal_Y']].values

# 출력하여 좌표 확인
print("RGB points:", rgb_points)
print("Thermal points:", thermal_points)

import cv2
import numpy as np

# RGB와 Thermal 좌표로 호모그래피 계산
H, _ = cv2.findHomography(rgb_points, thermal_points, method=cv2.RANSAC)

# 계산된 호모그래피 출력
print("Homography matrix:\n", H)

# 두 카메라 캡처 (예: RGB 카메라와 Thermal 카메라)
cap_rgb = cv2.VideoCapture(1)  # RGB 카메라 (0번 카메라)
cap_thermal = cv2.VideoCapture(0)  # Thermal 카메라 (1번 카메라)

while True:
    # RGB 이미지 캡처
    ret_rgb, frame_rgb = cap_rgb.read()
    if not ret_rgb:
        break  # 카메라 프레임이 없으면 종료

    # Thermal 이미지 캡처
    ret_thermal, frame_thermal = cap_thermal.read()
    if not ret_thermal:
        break  # 카메라 프레임이 없으면 종료

    # 호모그래피 매트릭스 적용
    if H is not None:
        aligned_rgb = cv2.warpPerspective(frame_rgb, H, (frame_thermal.shape[1], frame_thermal.shape[0]))

        # RGB 이미지와 Thermal 이미지 정합
        result_image = cv2.addWeighted(aligned_rgb, 0.5, frame_thermal, 0.5, 0)
        # 크기를 절대적으로 (width=800, height=600)으로 변경
        resized_image = cv2.resize(result_image, (800, 600), interpolation=cv2.INTER_LINEAR)

        # 정합된 이미지 출력
        cv2.imshow("Aligned Image", result_image)
        cv2.imshow('resized', resized_image)

    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 카메라 리소스 해제
cap_rgb.release()
cap_thermal.release()
cv2.destroyAllWindows()