import cv2

# 카메라 열기 (기본 카메라 사용)
cap = cv2.VideoCapture(0)

# 카메라가 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

# 프레임 크기 가져오기
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"프레임 크기: {frame_width}x{frame_height}")

# 카메라 해제
cap.release()
cv2.destroyAllWindows()
