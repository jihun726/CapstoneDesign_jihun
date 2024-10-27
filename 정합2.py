import cv2

# 카메라 설정
flir_camera = cv2.VideoCapture(0)  # 열화상 카메라
rgb_camera = cv2.VideoCapture(1)   # RGB 카메라

# 패턴 크기 설정
shape = (4, 11)  # 예: 7x7 도트 패턴

while True:
    ret_flir, flir_frame = flir_camera.read()
    ret_rgb, rgb_frame = rgb_camera.read()

    if ret_flir and ret_rgb:
        flir_copy = flir_frame.copy()
        rgb_copy = rgb_frame.copy()

        # 열화상 도트 패턴 인식
        ret_flir, corners_t = cv2.findCirclesGrid(flir_frame, shape)
        if ret_flir:
            cv2.drawChessboardCorners(flir_copy, shape, corners_t, ret_flir)

            # RGB 도트 패턴 인식
            ret_rgb, corners_r = cv2.findCirclesGrid(rgb_frame, shape)
            if ret_rgb:
                cv2.drawChessboardCorners(rgb_copy, shape, corners_r, ret_rgb)

                # 호모그래피 계산
                H, _ = cv2.findHomography(corners_r, corners_t, method=cv2.RANSAC)
                if H is not None:
                    img_warp = cv2.warpPerspective(rgb_frame, H, (240, 180))
                    result_image = cv2.addWeighted(img_warp, 0.5, flir_copy, 0.5, 0)
                    cv2.imshow('result', result_image)
                else:
                    print("호모그래피 계산 실패")
            else:
                print("RGB 도트 패턴 인식 실패")
        else:
            print("열화상 도트 패턴 인식 실패")

        # 화면에 표시
        cv2.imshow("flir", flir_copy)
        cv2.imshow("rgb", rgb_copy)
        cv2.imshow("RGB_1", ret_rgb)
        cv2.imshow("FLIR_1", ret_flir)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
flir_camera.release()
rgb_camera.release()
cv2.destroyAllWindows()
