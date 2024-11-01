import cv2
import numpy as np

# 원형 그리드 크기
pattern_size = (4, 11)

# 이미지 수
num_images = 25

for i in range(1, num_images + 1):
    img_path = f'rgb_images/{i:03}.png'
    img = cv2.imread(img_path)

    if img is None:
        print(f"Error loading image: {img_path}")
        continue

    # 원본 이미지 표시
    cv2.imshow(f'Original RGB Image {i}', img)

    # 그레이스케일 변환 및 반전
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inverted_gray = 255 - gray  # 이미지 반전

    # 대비 조정
    adjusted = cv2.convertScaleAbs(gray, alpha=2.0, beta=-100)

    # 이진화 처리
    _, binary = cv2.threshold(adjusted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 패턴 인식
    ret, corners = cv2.findCirclesGrid(binary, pattern_size, None, cv2.CALIB_CB_SYMMETRIC_GRID)

    if ret:
        cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow(f'Detected Pattern in RGB Image {i}', img)
    else:
        print(f"Pattern not found in RGB Image {i}.")

    # 반전된 이미지 표시
    cv2.imshow(f'Inverted RGB Image {i}', inverted_gray)

    # 키 입력 대기
    cv2.waitKey(5000)

cv2.destroyAllWindows()
