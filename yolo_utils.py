import cv2
import numpy as np
import darknet  # YOLO를 사용하는 라이브러리
import cv2

def get_contour_from_yolo(frame, network, class_names):
    """
    YOLO를 사용하여 가장 큰 물체의 윤곽선을 추출합니다.
    YOLO는 물체의 경계 상자를 반환하므로, 이를 이용해 윤곽선을 추정합니다.

    :param frame: 입력 이미지 (OpenCV로 캡처한 이미지)
    :param network: YOLO 네트워크 객체 (darknet에서 로드한 네트워크)
    :param class_names: YOLO 클래스 이름 리스트
    :return: 가장 큰 물체의 윤곽선으로 추정된 좌표들의 배열 (Nx2)
    """
    detections = detect_objects(frame, network, class_names)
    
    if not detections:
        raise ValueError("물체가 인식되지 않았습니다.")

    largest_contour = None
    max_area = 0

    for label, confidence, bbox in detections:
        x, y, w, h = bbox
        
        # 경계 상자의 넓이 계산
        area = w * h
        
        # 가장 큰 넓이의 물체 선택
        if area > max_area:
            max_area = area
            
            # 경계 상자의 좌표로부터 윤곽선을 추정 (사각형 윤곽선)
            top_left = (int(x - w/2), int(y - h/2))
            top_right = (int(x + w/2), int(y - h/2))
            bottom_right = (int(x + w/2), int(y + h/2))
            bottom_left = (int(x - w/2), int(y + h/2))
            
            largest_contour = np.array([top_left, top_right, bottom_right, bottom_left])
    
    # 가장 큰 물체의 윤곽선을 반환
    return largest_contour if largest_contour is not None else None


def load_yolo_model(config_file, data_file, weights_file):
    """
    YOLO 모델을 로드하는 함수입니다.

    :param config_file: YOLO 설정 파일 경로
    :param data_file: YOLO 데이터 파일 경로
    :param weights_file: YOLO 가중치 파일 경로
    :return: YOLO 네트워크 객체, 클래스 이름 리스트, 클래스 색상 리스트
    """
    network, class_names, class_colors = darknet.load_network(
        config_file=config_file,
        data_file=data_file,
        weights=weights_file
    )
    return network, class_names, class_colors

def detect_objects(frame, network, class_names):
    """
    YOLO를 사용하여 프레임 내에서 객체를 감지합니다.

    :param frame: 입력 이미지
    :param network: YOLO 네트워크 객체
    :param class_names: YOLO 클래스 이름 리스트
    :return: 감지된 객체의 리스트 (라벨, 신뢰도, 경계 상자)
    """
    darknet_image = darknet.make_image(darknet.network_width(network),
                                       darknet.network_height(network), 3)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb,
                               (darknet.network_width(network),
                                darknet.network_height(network)),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=0.5)
    darknet.free_image(darknet_image) # 이미지 메모리 해제
    
    return detections # 객체에 대한 정보(클래스 라벨, 신뢰도, 경계 상자 좌표)를 포함
