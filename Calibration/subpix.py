import numpy as np
import cv2

rgb = cv2.VideoCapture(2)
rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

shape = (4,11)

while True:
    key = cv2.waitKey(33) & 0xff
    rgb_ret, rgb_frame = rgb.read()

    if rgb_ret:
        cv2.imshow("RGB", rgb_frame)
        rgb_copy = rgb_frame.copy()

        rgb_gray = cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2GRAY)

        b,g,r = cv2.split(rgb_copy)
        cv2.imshow("rgb_gray", rgb_gray)
        g_inv = cv2.bitwise_not(g)
        cv2.imshow("g_inv", g_inv)

        ret_g, corners = cv2.findCirclesGrid(g, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        rgb_f = cv2.findCirclesGrid(g)
        cv2.imshow("fff", rgb_f)

    if ret_g == True:
        corners3 = corners
        corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
        cv2.drawChessboardCorners(rgb_copy, shape, corners3, ret_g)
        cv2.imshow("Detected Circles Grid", rgb_copy)

        # 보정된 코너 그리기
        cv2.drawChessboardCorners(frame, pattern_size, centers2, ret)

        # 결과를 별도의 창에 표시
        result_frame = frame.copy()
        cv2.imshow('Detected Circles Grid', result_frame)


    # 'q' 키를 눌러 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()