import cv2
from yolo_utils import *
from grip_utils import *

def get_frame_from_camera():
    """
    카메라로부터 실시간 프레임을 가져옵니다.

    :return: 캡처된 프레임 이미지
    """
    cap = cv2.VideoCapture(0)  # 디폴트 카메라 (0번 장치) 열기
    if not cap.isOpened():
        raise ValueError("카메라를 열 수 없습니다.")

    ret, frame = cap.read()  # 프레임 읽기
    if not ret:
        raise ValueError("프레임을 읽을 수 없습니다.")
    
    cap.release()  # 카메라 장치 해제
    return frame

def draw_grip_points(frame, points):
    """
    입력된 프레임 위에 파지 지점을 표시합니다.

    :param frame: 입력 이미지
    :param points: 파지 지점 좌표들의 리스트 [(x1, y1), (x2, y2), (x3, y3)]
    :return: 지점이 표시된 이미지
    """
    for point in points:
        cv2.circle(frame, tuple(point), 10, (0, 255, 0), -1)  # 초록색 원으로 표시
    return frame

if __name__ == "__main__":
    # YOLO 모델 초기화
    network, class_names, class_colors = load_yolo_model(
        config_file="yolov4.cfg",
        data_file="coco.data",
        weights_file="yolov4.weights"
    )
    
    frame = get_frame_from_camera()  # 카메라에서 프레임 가져오기

    try:
        contour = get_contour_from_yolo(frame, network, class_names)  # YOLO로 윤곽선 추출
        center_of_mass = calculate_center_of_mass(contour)  # 무게중심 계산

        best_grip_points = find_best_grip_points(contour, center_of_mass)
        print("최적의 파지 지점:", best_grip_points)

        # 파지 지점을 화면에 그리기
        frame_with_points = draw_grip_points(frame, best_grip_points)
        
        # 결과를 화면에 표시
        cv2.imshow("Grip Points", frame_with_points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except ValueError as e:
        print(e)
