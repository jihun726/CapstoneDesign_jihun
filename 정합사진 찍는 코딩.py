
import cv2
import os

# 저장할 디렉토리 설정
save_dir_rgb = 'C:/Users/김지훈/Desktop/캡스톤/rgb_calibration_images/'
save_dir_thermal = 'C:/Users/김지훈/Desktop/캡스톤/thermal_calibration_images/'

# 디렉토리 생성 (존재하지 않을 경우)
os.makedirs(save_dir_rgb, exist_ok=True)
os.makedirs(save_dir_thermal, exist_ok=True)

# RGB 및 열화상 카메라 초기화
cap_thermal = cv2.VideoCapture(1, cv2.CAP_DSHOW)       # 외부 RGB 카메라 장치 번호 확인 필요
cap_rgb = cv2.VideoCapture(2, cv2.CAP_DSHOW)   # 열화상 카메라 장치 번호 확인 필요

# 열화상 카메라 해상도 설정 (160x122로 설정)
cap_thermal.set(cv2.CAP_PROP_FRAME_WIDTH, 120)
cap_thermal.set(cv2.CAP_PROP_FRAME_HEIGHT, 100)

# 카메라가 열리지 않았을 경우 오류 출력
if not cap_rgb.isOpened():
    print("Error: Could not open RGB camera.")
    exit()
if not cap_thermal.isOpened():
    print("Error: Could not open thermal camera.")
    exit()

# 사진 번호 초기화
photo_count = 0

while cap_rgb.isOpened() and cap_thermal.isOpened():
    # 각 카메라로부터 프레임 읽기
    ret_rgb, frame_rgb = cap_rgb.read()
    ret_thermal, frame_thermal = cap_thermal.read()

    # 카메라로부터 프레임 읽기에 실패한 경우 오류 출력 후 루프의 다음 반복으로 이동
    if not ret_rgb or not ret_thermal:
        print("Warning: Failed to grab frame from one or both cameras.")
        continue

    # 열화상 이미지 컬러맵 적용 (COLORMAP_JET 또는 COLORMAP_HOT 사용)
    frame_thermal_colored = cv2.applyColorMap(frame_thermal, cv2.COLORMAP_TWILIGHT)

    # 화면에 현재 프레임 표시 (RGB 및 컬러맵이 적용된 열화상)
    cv2.imshow("RGB Camera", frame_rgb)
    cv2.imshow("Thermal Camera (Colored)", frame_thermal_colored)

    # 열화상 이미지 디노이즈 적용 (노이즈 제거)
    frame_thermal_denoised = cv2.fastNlMeansDenoising(frame_thermal, None, h=10, templateWindowSize=7, searchWindowSize=21)

    # 열화상 이미지 업스케일링 (2배 확대, INTER_CUBIC 보간법 사용)
    upscale_factor = 2
    width = int(frame_thermal_denoised.shape[1] * upscale_factor)
    height = int(frame_thermal_denoised.shape[0] * upscale_factor)
    frame_thermal_upscaled = cv2.resize(frame_thermal_denoised, (width, height), interpolation=cv2.INTER_CUBIC)

    # 화면에 현재 프레임 표시 (RGB 및 디노이즈 후 업스케일링된 열화상)
    cv2.imshow("RGB Camera", frame_rgb)
    cv2.imshow("Thermal Camera (Denoised & Upscaled)", frame_thermal_upscaled)

    # 'c' 키를 누르면 사진 저장
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        # RGB와 업스케일링된 열화상 이미지 저장
        rgb_save_path = os.path.join(save_dir_rgb, f"rgb_calib_{photo_count}.jpg")
        thermal_save_path = os.path.join(save_dir_thermal, f"thermal_calib_{photo_count}.jpg")

        cv2.imwrite(rgb_save_path, frame_rgb)
        cv2.imwrite(thermal_save_path, frame_thermal_upscaled)

        print(f"Saved RGB image to {rgb_save_path}")
        print(f"Saved Denoised & Upscaled Thermal image to {thermal_save_path}")

        # 사진 번호 증가
        photo_count += 1

    # 'q' 키를 누르면 종료
    elif key == ord('q'):
        break

# 자원 해제
cap_rgb.release()
cap_thermal.release()
cv2.destroyAllWindows()