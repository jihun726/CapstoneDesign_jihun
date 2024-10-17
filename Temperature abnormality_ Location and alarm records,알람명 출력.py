import cv2
import numpy as np
import datetime
import geocoder
from geopy.geocoders import Nominatim  # Geopy 라이브러리에서 Nominatim 서비스 사용

# 열화상 카메라 인덱스 설정 (예: 외부 열화상 카메라 인덱스 1)
thermal_camera_index = 1

# 열화상 카메라 비디오 캡처 객체 생성
thermal_cap = cv2.VideoCapture(thermal_camera_index)

# 카메라가 정상적으로 열렸는지 확인
if not thermal_cap.isOpened():
    print("열화상 카메라를 열 수 없습니다.")
    exit()

# 창 이름 설정 및 창 크기를 자유롭게 조절할 수 있는 플래그 설정
window_name = 'Thermal Temperature - Real-Time'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하게 설정

# 온도 범위 설정 (열화상 카메라에 따라 조정)
min_temp = 20  # 최소 온도 (예: 20도)
max_temp = 40  # 최대 온도 (예: 40도)
threshold_temp = 38  # 특정 온도 임계값 (예: 38도)

# 온도 초과 시 기록을 위한 리스트 초기화
temperature_records = []

# Geopy에서 위치 정보를 얻기 위한 Nominatim 객체 생성
geolocator = Nominatim(user_agent="thermal_camera_app")

# 현재 GPS 위치를 가져오는 함수
def get_current_location():
    try:
        g = geocoder.ip('me')  # IP 주소를 기반으로 위치 정보 가져오기
        if g.latlng:
            return g.latlng
        else:
            print("GPS 정보를 가져올 수 없습니다. 기본 위치 반환")
            return (0.0, 0.0)  # 위치 정보가 없을 경우 기본 좌표 반환
    except Exception as e:
        print(f"GPS 위치를 가져오는 중 오류 발생: {e}")
        return (0.0, 0.0)

# GPS 좌표를 한국식 주소로 변환하고 순서를 정렬하는 함수
def get_korean_address(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language='ko')  # Reverse Geocoding 수행 (한국어로 주소 반환)
        if location:
            # 주소를 ", "로 분리하고 요소 분류
            address_components = location.address.split(', ')
            
            # 주소를 구성 요소로 분리하여 순서 맞추기 (도/광역시, 시/군/구, 동/면/리, 세부 주소)
            province = city = district = detail = ""

            for component in address_components:
                # 도, 광역시 찾기
                if "도" in component or "광역시" in component:
                    province = component
                # 시, 군, 구 찾기
                elif "시" in component or "군" in component or "구" in component:
                    city = component
                # 동, 면, 리 찾기
                elif "동" in component or "면" in component or "리" in component:
                    district = component
                # 나머지는 세부 주소로 처리
                else:
                    detail += f"{component} "

            # 한국식 주소 형식으로 재구성
            korean_address = f"{province} {city} {district} {detail.strip()}"
            return korean_address.strip()
        else:
            return "주소를 찾을 수 없습니다."
    except Exception as e:
        print(f"주소 변환 중 오류 발생: {e}")
        return "주소 변환 오류"

while True:
    ret, frame = thermal_cap.read()
    if not ret:
        print("프레임을 읽어올 수 없습니다.")
        break

    # 프레임을 그레이스케일로 변환 (열화상 데이터는 기본적으로 온도를 밝기 값으로 표현)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 픽셀 값을 온도로 변환
    temp_frame = np.interp(gray_frame, (0, 255), (min_temp, max_temp))

    # 온도 프레임을 컬러맵 적용하여 시각적으로 변환
    temp_colormap = cv2.applyColorMap(cv2.convertScaleAbs(temp_frame, alpha=255 / max_temp), cv2.COLORMAP_JET)

    # 최고 온도 위치 및 온도 표시
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(temp_frame)

    # 특정 온도 초과 감지 및 위치 표시
    if max_val >= threshold_temp:
        # 현재 시간 기록
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # GPS 위치 정보 가져오기
        gps_location = get_current_location()
        if gps_location != (0.0, 0.0):
            korean_address = get_korean_address(gps_location[0], gps_location[1])  # 좌표를 한국식 주소로 변환
        else:
            korean_address = "위치 정보를 가져올 수 없음"

        # 온도 초과 기록 저장 (콘솔에만 출력, 화면에 출력하지 않음)
        temperature_records.append((current_time, max_loc, max_val, korean_address))

        # 콘솔에 경고 메시지 출력
        print(f"ALERT! 온도가 {threshold_temp}도 이상입니다.")
        print(f"시간: {current_time}")
        print(f"위치: {korean_address}")
        print(f"온도: {max_val:.2f}C\n")

    # 결과 영상 표시 (카메라 화면에는 표시하지 않음)
    cv2.imshow(window_name, temp_colormap)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 열화상 카메라 해제 및 창 닫기
thermal_cap.release()
cv2.destroyAllWindows()

# 온도 초과 기록 출력 (종료 시 기록된 정보 콘솔에 출력)
print("\n온도 초과 기록:")
for record in temperature_records:
    print(f"시간: {record[0]}, 위치: {record[3]}, 온도: {record[2]:.2f}C")
