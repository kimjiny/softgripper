import numpy as np
import darknet  # YOLO를 사용하는 라이브러리
import cv2
import RPi.GPIO as GPIO
import time

# GPIO 핀 설정
SERVO_PIN = 18  # 서보 모터가 연결된 GPIO 핀 번호
FREQUENCY = 50  # PWM 주파수 (서보 모터의 일반적인 주파수는 50Hz)

# YOLO 모델을 초기화
network, class_names, class_colors = darknet.load_network(
    config_file="yolov4.cfg",
    data_file="coco.data",
    weights="yolov4.weights"
)

# 카메라에서 실시간으로 프레임을 가져오는 함수
def get_frame_from_camera():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame

# YOLO를 사용하여 물체를 인식하고, 아웃라인을 반환
def detect_objects(frame, network, class_names):
    # YOLO로 이미지에서 물체 탐지
    darknet_image = darknet.make_image(darknet.network_width(network),
                                       darknet.network_height(network), 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #이미지 처리방식 변경 opencv에서 YOLO로 변경
    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(network),
                                darknet.network_height(network)),
                               interpolation=cv2.INTER_LINEAR) #프레임을 네트워크 규격대로 보간법을 사용하여 리사이즈

    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes()) # 리사이즈드 된 이미지를 바이트로 변환하기
    # 신뢰도 0.5이상의 객체만 값을 탐지한다. label(라벨링) confidence(신뢰도) bbox(윤곽선)의 값으로 반환
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5) 
    darknet.free_image(darknet_image) #메모리 관리를 위해서 기존의 기억중인 이미지 해제
    
    # 탐지된 물체 중 가장 큰 것의 경계 상자 반환
    max_area = 0
    selected_bbox = None
    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        area = w * h
        if area > max_area:
            max_area = area
            selected_bbox = bbox
    
    if selected_bbox:
        x, y, w, h = selected_bbox
        outline = np.array([[x - w/2, y - h/2], [x + w/2, y - h/2],
                            [x + w/2, y + h/2], [x - w/2, y + h/2]])
        return outline
    else:
        return None

# 물체의 무게중심 계산
# m00 : 면적을 나타내는 0차 모멘트로, 윤곽선이 차지하는 전체 픽셀 수
# m10 : x축에 대한 1차 모멘트
# m01 : y축에 대한 1차 모멘트
def calculate_center_of_mass(outline):
    M = cv2.moments(outline)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return (cx, cy)

# 파지 지점 찾기
def find_grip_points(outline, center_of_mass, num_points=3):
    outline_points = np.array(outline)
    
    # 무게중심에서 각 점까지의 벡터를 계산하고, 각도를 구함
    vectors = outline_points - center_of_mass
    angles = np.arctan2(vectors[:, 1], vectors[:, 0])
    
    # 각도를 오름차순으로 정렬하고, 그에 따른 인덱스를 얻음
    sorted_indices = np.argsort(angles)
    sorted_angles = angles[sorted_indices]
    sorted_outline_points = outline_points[sorted_indices]
    
    best_set = None
    min_angle_diff = np.inf
    
    # 각 점에서 시작하여 120도 간격에 가장 가까운 세 점을 선택
    for i in range(len(sorted_angles)):
        for j in range(i + 1, len(sorted_angles)):
            for k in range(j + 1, len(sorted_angles)):
                angle_diff_1 = np.abs(sorted_angles[j] - sorted_angles[i])
                angle_diff_2 = np.abs(sorted_angles[k] - sorted_angles[j])
                angle_diff_3 = np.abs((sorted_angles[i] + 2 * np.pi) - sorted_angles[k])
                
                angle_diffs = np.array([angle_diff_1, angle_diff_2, angle_diff_3])
                angle_diffs = np.sort(angle_diffs)
                
                # 120도 (2 * pi / 3) 간격에 가장 가까운 세 점을 선택
                if np.all(np.abs(angle_diffs - 2 * np.pi / 3) < np.deg2rad(10)):
                    angle_diff_sum = np.sum(np.abs(angle_diffs - 2 * np.pi / 3))
                    if angle_diff_sum < min_angle_diff:
                        min_angle_diff = angle_diff_sum
                        best_set = [sorted_outline_points[i], sorted_outline_points[j], sorted_outline_points[k]]
    
    if best_set is not None:
        return np.array(best_set)
    else:
        raise ValueError("유효한 파지 지점을 찾을 수 없습니다.")

def calculate_rotation_angle(current_finger_positions, best_set, center_of_mass):
    """
    현재 손가락 위치와 목표 파지 지점들을 비교하여,
    그리퍼를 회전시켜야 하는 각도를 계산합니다.

    :param current_finger_positions: 현재 손가락들의 좌표 배열 (3개의 점)
    :param best_set: 목표 파지 지점들의 좌표 배열 (3개의 점)
    :param center_of_mass: 무게중심 좌표
    :return: 회전 각도 (도 단위)
    """
    # 첫 번째 손가락과 첫 번째 목표 파지 지점에 대한 벡터 계산
    current_vector = current_finger_positions[0] - center_of_mass
    target_vector = best_set[0] - center_of_mass

    # 현재 벡터와 목표 벡터의 각도 계산
    current_angle = np.arctan2(current_vector[1], current_vector[0])
    target_angle = np.arctan2(target_vector[1], target_vector[0])

    # 회전 각도 계산
    rotation_angle = target_angle - current_angle

    # 회전 각도를 도 단위로 변환
    rotation_angle_degrees = np.rad2deg(rotation_angle)

    return rotation_angle_degrees


def rotate_motor(degree):
    pwm = setup_gpio()
    
    try:
        # 각도(degree)를 PWM 듀티 사이클로 변환
        duty_cycle = (0.05 * FREQUENCY) + (0.19 * FREQUENCY * (degree / 180.0))
        
        # 서보 모터를 회전
        pwm.ChangeDutyCycle(duty_cycle)
        print(f"모터를 {degree}도 만큼 회전합니다.")
        
        # 회전 후 약간의 지연 시간을 주어 모터가 안정되도록 함
        time.sleep(1)
        
    finally:
        # 서보 모터를 정지하고 GPIO 설정을 정리
        pwm.stop()
        GPIO.cleanup()
        print("모터 회전이 완료되었습니다.")

# 메인 로직
if __name__ == "__main__":
    # 현재 손가락 위치 (임의의 예시 좌표)
    current_finger_positions = np.array([
        [10, 0],
        [5, 10],
        [-5, 10]
    ])
    frame = get_frame_from_camera()
    outline = detect_objects(frame, network, class_names)

    if outline is not None:
        center_of_mass = calculate_center_of_mass(outline)
        try:
            selected_points = find_grip_points(outline, center_of_mass, num_points=3)
            rotate_motor(calculate_rotation_angle(current_finger_positions, selected_points, center_of_mass))
            # 정확한 힘제어를 위해서 3개의 지점에 3개의 손가락이 동시에 닿을 수 있도록 시간을 제어하는 함수 추가하기

            print("무게중심:", center_of_mass)
            print("선택된 파지점:", selected_points)
        except ValueError as e:
            print(e)
    else:
        print("물체가 인식되지 않았습니다.")