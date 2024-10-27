import numpy as np
import cv2

rgb = cv2.VideoCapture(1)

#rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


flir = cv2.VideoCapture(0)
#flir.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
#flir.set(cv2.CAP_PROP_FRAME_HEIGHT, 256)


shape = (4,11)

while True:
    key = cv2.waitKey(33) & 0xff
    rgb_ret, rgb_frame = rgb.read()
    flir_ret, flir_frame = flir.read()


    if rgb_ret:
        cv2.imshow("RGB", rgb_frame)
        cv2.imshow("flir", flir_frame)

        rgb_copy = rgb_frame.copy()
        flir_copy = flir_frame.copy()

        rgb_gray = cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2GRAY)

        b,g,r = cv2.split(rgb_copy)
        cv2.imshow("rgb_gray", rgb_gray)
        g_inv = cv2.bitwise_not(g)
        cv2.imshow("g_inv", g_inv)

        ret_g, corners = cv2.findCirclesGrid(g, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret_g == True:
            corners3 = corners
            #corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
            cv2.drawChessboardCorners(rgb_copy, shape, corners3, ret_g)
            cv2.imshow("result_g", rgb_copy)

        ret_g_inv, corners = cv2.findCirclesGrid(g_inv, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret_g_inv == True:
            corners2 = corners
            # corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
            cv2.drawChessboardCorners(rgb_copy, shape, corners2, ret_g_inv)
            cv2.imshow("result_g_inv", rgb_copy)

        ret_flir, corners = cv2.findCirclesGrid(flir_frame, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret_flir == True:
            corners2 = corners
            # corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
            cv2.drawChessboardCorners(flir_copy, shape, corners2, ret_flir)
            cv2.imshow("result_flir", flir_copy)


    if key == ord('q'):
        break

    if (key&0xff) == ord('s'):
        if res is not None:
            file_name = 'rgbt_images_for_registration/image_{}.png'.format(str(count).zfill(5))
            cv2.imwrite(file_name, res)
            count += 1

    elif (key&0xff) == ord('c'):
        cv2.destroyAllWindows()
        break


cv2.destroyAllWindows()
