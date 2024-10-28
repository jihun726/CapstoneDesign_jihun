import numpy as np
import cv2

rgb = cv2.VideoCapture(1)
rgb.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
rgb.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

flir = cv2.VideoCapture(2)
flir.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
flir.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)

shape = (4,11)

image_counter = 1

while True:
    key = cv2.waitKey(33) & 0xff
    flir_ret, flir_frame = flir.read()
    rgb_ret, rgb_frame = rgb.read()

    if rgb_ret and flir_ret:
        cv2.imshow("RGB", rgb_frame)
        cv2.imshow("flir", flir_frame)
        print(flir_frame.shape)
        rgb_copy = rgb_frame.copy()
        flir_copy = flir_frame.copy()

        rgb_gray = cv2.cvtColor(rgb_copy, cv2.COLOR_BGR2GRAY)

        b,g,r = cv2.split(rgb_copy)
        cv2.imshow("rgb_gray", rgb_gray)
        g_inv = cv2.bitwise_not(g)
        cv2.imshow("g_inv", g_inv)

        ret_g, corners = cv2.findCirclesGrid(g, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        '''
        if ret_g == True:
            corners3 = corners
            #corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
            cv2.drawChessboardCorners(rgb_copy, shape, corners3, ret_g)
            cv2.imshow("result_g", rgb_copy)
        '''
        ret_g_inv, corners_r = cv2.findCirclesGrid(g_inv, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)

        if ret_g_inv == True:
            # corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
            cv2.drawChessboardCorners(rgb_copy, shape, corners_r, ret_g_inv)
            cv2.imshow("result_g_inv", rgb_copy)

        ret_flir, corners_t = cv2.findCirclesGrid(flir_frame, shape, None, flags=cv2.CALIB_CB_ASYMMETRIC_GRID)
        if ret_flir == True:
            # corners2 = cv2.cornerSubPix(g_inv, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0))
            cv2.drawChessboardCorners(flir_copy, shape, corners_t, ret_flir)
            cv2.imshow("result_flir", flir_copy)

            H , _ = cv2.findHomography(corners_r, corners_t, method = cv2.RANSAC)
            img_warp = cv2.warpPerspective(rgb_frame, H, (160, 120))
            result_image = cv2.addWeighted(img_warp, 0.5, flir_copy, 0.5, 0)
            cv2.imshow('result', result_image)
            #cv2.imwrite('result_registration_image', result_image)
            #cv2.waitkey(0)
            '''
            if H is not None:
                print(H)
            '''
            # Homography가 성공적으로 계산된 경우에만 진행
            if H is not None:
                # flir_copy의 크기를 img_warp에 맞게 조정
                flir_resized = cv2.resize(flir_copy, (img_warp.shape[1], img_warp.shape[0]))

                # 두 이미지를 합성
                result_image = cv2.addWeighted(img_warp, 0.5, flir_resized, 0.5, 0)
                cv2.imshow('result', result_image)
        if ret_flir and ret_g_inv:
            if len(corners_r) == len(corners_t) and len(corners_r) >= 4:
                H, _ = cv2.findHomography(corners_r, corners_t, method=cv2.RANSAC)
                if H is not None:
                    img_warp = cv2.warpPerspective(rgb_frame, H, (160, 120))
                    result_image = cv2.addWeighted(img_warp, 0.5, flir_copy, 0.5, 0)
                    cv2.imshow('result', result_image)
                    # 원하는 크기로 조정 (예: 800x600)
                    result_image_resized = cv2.resize(result_image, (320, 240))
                    cv2.imshow('result_1', result_image_resized)
                else:
                    print("호모그래피 계산 실패")
            else:
                print("코너 개수가 맞지 않습니다:", len(corners_r), len(corners_t))

    if key == ord('q'):
        break
    

    # 's' 키를 눌렀을 때만 저장
    if ret_flir and ret_g_inv and cv2.waitKey(1) & 0xFF == ord('s'):
        filename = f'registration_images/{image_counter:03d}.png'  # 파일 이름 생성
        cv2.imwrite(filename, result_image_resized)
        print(f'{filename} 저장 완료')  # 저장 완료 메시지 출력
        image_counter += 1  # 카운터 증가

    if ret_flir and ret_g_inv and cv2.waitKey(1) & 0xFF == ord('t'):
        filename = f'thermal_images/{image_counter:03d}.png'  # 파일 이름 생성
        cv2.imwrite(filename, flir_copy)
        print(f'{filename} 저장 완료')  # 저장 완료 메시지 출력
        image_counter += 1

    if ret_flir and ret_g_inv and cv2.waitKey(1) & 0xFF == ord('t'):
        filename = f'rgb_images/{image_counter:03d}.png'  # 파일 이름 생성
        cv2.imwrite(filename, rgb_copy)
        print(f'{filename} 저장 완료')  # 저장 완료 메시지 출력
        image_counter += 1

    if (key&0xff) == ord('c'):
        cv2.destroyAllWindows()
        break


cv2.destroyAllWindows()
