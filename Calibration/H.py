import cv2
import numpy as np

# 원형 그리드 크기
pattern_size = (4, 11)  # 예: 4x11 그리드

# 호모그래피 계산을 위한 포인트 저장 리스트
rgb_points = []
thermal_points = []

# 이미지 수
num_images = 25  # 각 이미지의 개수

# RGB 이미지 처리
for i in range(1, num_images + 1):
    img_path_r = f'rgb_images/{i:03}.png'  # 001, 002, ... 형식
    img_r = cv2.imread(img_path_r)
    img_r_copy = img_r.copy()
    gray = cv2.cvtColor(img_r_copy, cv2.COLOR_BGR2GRAY)
    g_inv = 255 - gray
    #cv2.imshow(f'RGB Image_1 {i}', g_inv)
    ret_g_inv, corners_r = cv2.findCirclesGrid(g_inv, pattern_size, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
    #print(corners_r)
    if ret_g_inv == True:
        # corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
        cv2.drawChessboardCorners(img_r_copy, pattern_size, corners_r, ret_g_inv)
        cv2.imshow("result_g_inv", img_r_copy)
        rgb_points.append(corners_r)

    img_path_t = f'thermal_images/{i:03}.png'  # 001, 002, ... 형식
    img_t = cv2.imread(img_path_t)
    img_t_copy = img_t.copy()
    ret_t, corners_t = cv2.findCirclesGrid(img_t_copy, pattern_size, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

    if ret_t == True:
        # corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
        cv2.drawChessboardCorners(img_t_copy, pattern_size, corners_t, ret_t)
        cv2.imshow("result_t", img_t_copy)
        thermal_points.append(corners_t)
        #print(corners_t)

    cv2.waitKey(33)
rgb_points_a = np.array(rgb_points)
thermal_points_a = np.array(thermal_points)

print(rgb_points)
print(thermal_points)

if rgb_points and thermal_points and len(rgb_points) == len(thermal_points):
    #H, _ = cv2.findHomography(rgb_points_a, thermal_points_a, method=cv2.RANSAC)
    H, _ = cv2.findHomography(rgb_points_a.reshape(-1, 2), thermal_points_a.reshape(-1, 2), method=cv2.RANSAC)
    print("Homography matrix:\n", H)

    if H is not None:
        im_dst = cv2.warpPerspective(img_r, H, (160, 120))

        result_image = cv2.addWeighted(im_dst, 0.5, img_t, 0.5, 0)
        cv2.imshow('registrated_img', result_image)
        cv2.waitKey(0)
