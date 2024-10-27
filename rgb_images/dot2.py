import numpy as np
import cv2

rgb = cv2.VideoCapture(1)
rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

flir = cv2.VideoCapture(0)

shape = (4, 11)  # 체스보드 또는 원형 그리드 크기 설정

while True:
    key = cv2.waitKey(33) & 0xff
    rgb_ret, rgb_frame = rgb.read()
    flir_ret, flir_frame = flir.read()

    if rgb_ret and flir_ret:
        cv2.imshow("RGB", rgb_frame)
        cv2.imshow("FLIR", flir_frame)

        rgb_copy = rgb_frame.copy()
        flir_copy = flir_frame.copy()

        rgb_gray = cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2GRAY)
        g = cv2.split(rgb_copy)[1]  # Green channel

        ret_g_inv, corners_r = cv2.findCirclesGrid(g, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

        if ret_g_inv:
            cv2.drawChessboardCorners(rgb_copy, shape, corners_r, ret_g_inv)
            cv2.imshow("result_g_inv", rgb_copy)

        ret_flir, corners_t = cv2.findCirclesGrid(flir_frame, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

        if ret_flir:
            cv2.drawChessboardCorners(flir_copy, shape, corners_t, ret_flir)
            cv2.imshow("result_flir", flir_copy)

            if corners_r and corners_t:
                H, _ = cv2.findHomography(corners_r, corners_t, method=cv2.RANSAC)
                img_warp = cv2.warpPerspective(rgb_frame, H, (flir_frame.shape[1], flir_frame.shape[0]))
                result_image = cv2.addWeighted(img_warp, 0.5, flir_copy, 0.5, 0)
                cv2.imshow('result', result_image)
                cv2.imwrite('result_registration_image.png', result_image)  # 확장자 추가

            if H is not None:
                print(H)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
