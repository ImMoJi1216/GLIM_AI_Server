import socket
import cv2
import numpy as np

def preprocess_image(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    # BGR에서 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 가우시안 블러 적용(노이즈 감소)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 그레이스케일 이미지의 적응형 스레시홀드 적용
    _, thresholded = cv2.threshold(blurred, 240, 255, cv2.THRESH_BINARY_INV)
    # 모폴로지 연산을 사용하여 그림자 제거
    kernel = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 이미지의 노이즈를 제거하기 위해 모폴로지 클로징 수행
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=2)
    # 검은 배경 이미지 생성
    black_background = np.zeros_like(image)
    # 원본 이미지에서 검은 배경에 해당 영역을 복사
    black_background[closed == 255] = image[closed == 255]

    return black_background


def determine_shape(contour):
    # 윤곽의 길이 확인
    if len(contour) < 3:
        return 'undefined shape'

    # 도형 근사화
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 도형의 꼭지점 수
    num_vertices = len(approx)

    # 각 도형에 대한 근사화된 꼭지점 수에 따라 도형 결정
    if num_vertices == 3:
        return 'triangle'
    elif num_vertices == 4:
        # 꼭지점이 4개일 때, 정사각형과 직사각형 구분
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.95 <= aspect_ratio <= 1.05:
            return 'foursquare'
        else:
            return 'rectangle'
    elif num_vertices == 5:
        return 'pentagon'
    elif num_vertices == 6:
        return 'hexagon'
    else:
        return 'circle'


def detect_shapes_and_colors(image_path):
    # 이미지 불러오기
    preprocessed_image = preprocess_image(image_path)

    # BGR에서 HSV로 변환
    hsvFrame = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2HSV)

    # 각 색상에 대한 범위 설정
    color_ranges = {
        'red': ([0, 100, 100], [10, 255, 255]),
        'orange': ([15, 100, 100], [25, 255, 255]),
        'yellow': ([25, 52, 72], [45, 255, 255]),
        'green': ([35, 52, 72], [80, 255, 255]),
        'blue': ([90, 80, 2], [120, 255, 255]),
        'pink': ([150, 40, 180], [170, 255, 255]),
        'purple': ([120, 40, 100], [150, 255, 255]),
        'brown': ([10, 100, 20], [20, 255, 200]),
        'sky_blue': ([80, 100, 100], [100, 255, 255]),
        # 'lavender': ([110, 38, 100], [130, 255, 255]),
        'gold': ([20, 100, 100], [30, 255, 255]),
        'silver': ([0, 0, 75], [180, 10, 150]),
        'cyan': ([85, 60, 60], [105, 255, 255]),
    }

    # 이미 처리된 영역을 저장할 배열
    processed_regions = np.zeros_like(preprocessed_image)

    # 전체 도형 카운트 변수
    total_shape_count = 0

    # 각 색상에 대한 영역 검출 및 표시
    for color, (lower, upper) in color_ranges.items():
        color_lower = np.array(lower, np.uint8)
        color_upper = np.array(upper, np.uint8)
        color_mask = cv2.inRange(hsvFrame, color_lower, color_upper)
        color_mask = cv2.dilate(color_mask, np.ones((5, 5), "uint8"))

        # 이미 처리된 영역을 마스킹
        color_mask = cv2.subtract(color_mask, processed_regions[:, :, 0])
        contours, hierarchy = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        max_area = 0  # 가장 큰 면적 초기화
        max_area_color = ''  # 가장 큰 면적을 가진 색상 초기화

        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            if area > 200:
                # 외곽선의 경계 확인
                x, y, w, h = cv2.boundingRect(contour)
                if x > 5 and y > 5 and x + w <  preprocessed_image.shape[1] - 5 and y + h < preprocessed_image.shape[
                    0] - 5:
                    # 모양 추출
                    shape_name = determine_shape(contour)

                    # 중심 좌표 계산
                    M = cv2.moments(contour)
                    if M['m00'] != 0.0:
                        x = int(M['m10'] / M['m00'])
                        y = int(M['m01'] / M['m00'])

                        # 색상 면적이 가장 큰 경우 업데이트
                        if area > max_area:
                            max_area = area
                            max_area_color = color

                        total_shape_count += 1
                        print(
                            f'{total_shape_count}. 도형 정보 - 색상: {max_area_color.upper()}, '
                            f'모양: {shape_name}, 좌표: ({x}, {y})')
                    total_result = color.upper() + "," + shape_name + "," + str(x) + "," + str(y)  # 결과치 압축

                    return total_result

def start_server():
    host = '10.10.21.118'
    port = 30000

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print("--------------------------------------------------------")
    print(f"서버가 IP : {host} / PORT : {port} 의 환경으로 실행중임")
    print("--------------------------------------------------------")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"중앙서버가 연결되었습니다 : {client_address}")

        while True:
            # 파일 정보 수신
            file_info = client_socket.recv(1024).decode('utf-8')
            if not file_info:
                break

            _, file_size, file_name = file_info.split(',')
            file_size = int(file_size)

            print(f"수신된 파일: {file_name}, 크기: {file_size} 바이트")

            # 파일 수신
            received_data = b""
            chunk_size = 8192 # 버퍼사이즈임 필요에따라 조절하면 쌉가능
            while True:
                data = client_socket.recv(chunk_size)
                if not data:
                    break
                received_data += data

                # 파일 전송이 완료되면 루프를 종료
                if len(received_data) == file_size:
                    break

            # 파일 저장
            save_path = "C:/Users/LMS23/Desktop/GLIM_AI_Save/" + file_name
            with open(save_path, 'wb') as file:
                file.write(received_data)

            print(f"파일 저장 완료: {save_path}")

            finish_analyze = detect_shapes_and_colors(save_path)

            print("결과값 : " + finish_analyze)
            print("\n")

            #client_socket.send("Type: Python,") +

if __name__ == '__main__':
    start_server()
