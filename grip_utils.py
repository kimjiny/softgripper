import numpy as np
import cv2
import math

def calculate_center_of_mass(contour):
    """
    윤곽선에서 무게중심을 계산합니다.

    :param contour: 물체의 윤곽선
    :return: 무게중심 좌표 (cx, cy)
    """
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
    else:
        cx, cy = 0, 0
    return (cx, cy)

def find_best_grip_points(contour, center_of_mass):
    """
    무게중심을 기준으로 120도 관계를 이루는 세 지점을 찾아, 
    이들의 무게중심으로부터의 거리합이 가장 짧은 집합을 반환합니다.

    :param contour: 물체의 윤곽선
    :param center_of_mass: 무게중심 좌표 (cx, cy)
    :return: 최적의 세 지점 좌표 [(x1, y1), (x2, y2), (x3, y3)]
    """
    points = contour  # 윤곽선의 모든 점들
    best_set = None
    min_distance_sum = float('inf') # 최소값을 찾기 위해 초기값을 매우 큰 값으로 설정
    
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            for k in range(j + 1, len(points)):
                p1, p2, p3 = points[i], points[j], points[k]
                
                # 벡터를 무게중심으로부터 계산
                vec_p1 = p1 - center_of_mass
                vec_p2 = p2 - center_of_mass
                vec_p3 = p3 - center_of_mass

                # 무게중심으로부터 각 지점까지의 거리
                d1 = np.linalg.norm(p1 - center_of_mass)
                d2 = np.linalg.norm(p2 - center_of_mass)
                d3 = np.linalg.norm(p3 - center_of_mass)
                
                # 각도 계산 (무게중심 기준)
                angle_12 = math.degrees(math.atan2(vec_p2[1], vec_p2[0]) - math.atan2(vec_p1[1], vec_p1[0]))
                angle_13 = math.degrees(math.atan2(vec_p3[1], vec_p3[0]) - math.atan2(vec_p1[1], vec_p1[0]))

                # 각도를 절대값으로 변환하고, 360도를 넘어가는 경우 보정
                angle_12 = abs(angle_12) % 360
                angle_13 = abs(angle_13) % 360

                # 각도 차이 계산 (120도 관계인지 확인)
                if 115 <= angle_12 <= 125 and 115 <= angle_13 <= 125:
                    distance_sum = d1 + d2 + d3
                    if distance_sum < min_distance_sum:
                        min_distance_sum = distance_sum
                        best_set = [p1, p2, p3]

    if best_set is not None:
        return best_set
    else:
        raise ValueError("120도 관계를 이루는 유효한 세 지점을 찾을 수 없습니다.")
