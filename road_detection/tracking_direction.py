import supervision as sv
import cv2
import torch
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import math
import os
import warnings
# FutureWarning 무시
warnings.simplefilter(action='ignore', category=FutureWarning)

'''
차량 객체를 탐지하고 추적하여 진행 방향을 설정하고 그 방향을 표시하는 모듈
'''
# arguments 설정
class arguments:
    def __init__(self, source_video_path, target_video_path=None, weight_path=None, confidence_threshold=0.4, iou_threshold=0.5):
        self.source_video_path = source_video_path
        self.weight_path = weight_path
        self.target_video_path = target_video_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def print_args(self):
        pass

# yolov5모델 로드를 위한 함수
def load_yolov5_model(weights_path):
    # Load the model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)

    if torch.cuda.is_available():
        model = model.to('cuda')  
        print("CUDA")
    else:
        print("CPU")

    model.eval()  # Set model to evaluation mode
    return model

class Perspective_transformer:
    '''
    원근변환 행렬 생성 클래스
    '''
    def __init__(self, source: np.ndarray, target:np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        # 원근 변환 행렬 계산
        self.m = cv2.getPerspectiveTransform(source, target)
    
    # tracking 되는 객체에 대한 bottom center 좌표를 변환
    def transform_points(self, points: np.array):
        if points.size == 0:
            return points
        
        # perspectiveTransform 입력 인자로 3채널 array가 필요
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1,2)
    
def find_boundary_intersection(x_min, y_min, x_max, y_max, vx, vy):
    """
    현재 프레임 기준의 바운딩 박스에서 방향 벡터를 이루는 직선과 접하는 바운딩박스 좌표 구하기

    Args:
    - x_min, y_min (float): Top-left corner of the bounding box.
    - x_max, y_max (float): Bottom-right corner of the bounding box.
    - vx, vy (float): Direction vector components.
    - cx, cy (float): Starting point of the vector (e.g., the center of the bounding box).

    Returns:
    - tuple: (x, y) intersection point or None if no intersection.
    """
    # 바운딩 박스 경계 설정
    left = x_min
    right = x_max
    top = y_min
    bottom = y_max

    cx = (right + left) / 2
    cy = (bottom + top) / 2

    # 방향 벡터에 따른 교점 계산
    intersections = []

    # 위쪽 변
    if vy != 0:
        t = (top - cy) / vy
        x = cx + t * vx
        if left <= x <= right and t > 0:
            intersections.append((x, top))

    # 아래쪽 변
    if vy != 0:
        t = (bottom - cy) / vy
        x = cx + t * vx
        if left <= x <= right and t > 0:
            intersections.append((x, bottom))

    # 왼쪽 변
    if vx != 0:
        t = (left - cx) / vx
        y = cy + t * vy
        if top <= y <= bottom and t > 0:
            intersections.append((left, y))

    # 오른쪽 변
    if vx != 0:
        t = (right - cx) / vx
        y = cy + t * vy
        if top <= y <= bottom and t > 0:
            intersections.append((right, y))

    # 가장 가까운 교점 반환
    if intersections:
        return min(intersections, key=lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5)
    return None

def main(args):

    #video에 대한 정보 - 높이 너비 fps 등등..
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)

    # yolov5 model load
    model = load_yolov5_model(args.weight_path)

    #tracking using supervision
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh = video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(
        resolution_wh = video_info.resolution_wh)

    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    label_annotator = sv.LabelAnnotator(text_scale=text_scale, text_thickness=thickness, text_position=sv.Position.BOTTOM_CENTER)
    
    trace_annotator = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    
    # 좌표 초기화 deque
    x_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))
    y_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))
    xyxy = defaultdict(list)
    
    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            # model infer (detect.py)
            frame = frame.cuda()
            result = model(frame)
            # yolov5 detect 반환 값에 대한 sv.Detections.from_yolov5 모듈 적용
            detections = sv.Detections.from_yolov5(result)
            # confidence 이상인 값만 추출
            detections = detections[detections.confidence > args.confidence_threshold]
            # nms
            detections = detections.with_nms(threshold=args.iou_threshold)
          
            # 객체 탐지 반환 값을 가지고 tracking
            '''
            detection은 객체 탐지 + 추적 클래스
            detections.tracker_id => 추적 객체 id
            detections.get_anchors_cooordinates(anchor=sv.Position.CENTER) => 추적 객체 좌표 반환 방식(예시에선 Center중심점 반환)
            '''
            detections = byte_track.update_with_detections(detections= detections)

            # 탐지된 객체의 bottom center 좌표
            # 궤적 방향 파악해서 어떤 좌표(center/bottom 등등)로 설정할 건지 정하기
            car_points = detections.get_anchors_coordinates(anchor=sv.Position.CENTER)
            
            # tracking id, (x,y) coordinates (float32)
            for tracker_id, [x, y], xyxy_point in zip(detections.tracker_id, car_points, detections.xyxy):
                xyxy[tracker_id].append(xyxy_point)
                x_coordinates[tracker_id].append(x)
                y_coordinates[tracker_id].append(y)
            
            for tracker_id in detections.tracker_id:
                # 최소 0.5초 이상 동안 탐지된 객체들만 tracking
                if len(y_coordinates[tracker_id]) < video_info.fps / 2: 
                    pass
                    #labels.append(f"#{tracker_id}")
                else:
                    # deque 자료구조에서 반환
                    x_coordinates_start = x_coordinates[tracker_id][-1] #가장 최근 프레임의 x_center 좌표
                    x_coordinates_end = x_coordinates[tracker_id][-5] #가장 최근 프레임의 y_center 좌표

                    y_coordinates_start = y_coordinates[tracker_id][-1]
                    y_coordinates_end = y_coordinates[tracker_id][-5]
                    
                    ######################## 방향 벡터를 이루는 직선과 바운딩박스가 접하는 지점 구하기 (필요시)
                    # 방향 벡터 계산 
                    # dx = x_coordinates_start - x_coordinates_end
                    # dy = y_coordinates_start - y_coordinates_end
                    
                    # x_min, y_min, x_max, y_max = xyxy[tracker_id][-1][0], xyxy[tracker_id][-1][1], xyxy[tracker_id][-1][2], xyxy[tracker_id][-1][3]
                    # # x2_min, y2_min, x2_max, y2_max = xyxy[tracker_id][-5][0], xyxy[tracker_id][-5][1], xyxy[tracker_id][-5][2], xyxy[tracker_id][-5][3]
                    # inter_point = find_boundary_intersection(x_min,y_min,x_max, y_max, dx, dy)
                    # if inter_point != None:
                    #     inter_point = tuple(map(int, inter_point))
                    #     cv2.circle(frame, inter_point,radius=4, color=(0,0,255), thickness=-1)
                    # else: pass
                    # angle = math.degrees(math.atan2(dy, dx))  # 각도 계산
                    # labels.append(f"#{tracker_id} {int(angle)} degree")
                    ################################

                    # 벡터 화살표 그리기
                    # 끝점에 화살표 생성
                    cv2.arrowedLine(
                        frame,
                        (int(x_coordinates_end), int(y_coordinates_end)),  # 시작점
                        (int(x_coordinates_start), int(y_coordinates_start)),  # 끝점
                        (0, 0, 255),                  # 화살표 색상 (빨간색)
                        2,                            # 선 두께 Thickness
                        tipLength=0.3                 # 화살촉 길이
                    )
                    
            labels = [
                f"#{tracker_id}" for tracker_id in detections.tracker_id
            ]
            annotated_frame = frame.copy()
            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # box 시각화 클래스
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections)
            # label 시각화 클래스
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels = labels)

            sink.write_frame(annotated_frame)
            cv2.imshow("annotated_frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    current_directory = os.getcwd()
    BASE = os.path.dirname(current_directory)
    source_video_path = BASE+r'\data\videos\test_video4.mp4'
    target_video_path = BASE+r'\results\road_detection\test_video4.mp4'

    weight_path = r'C:\Users\QBIC\Desktop\workspace\car_lane_detect\Yolov5\weights\1280720_3rd.pt' 
    args = arguments(source_video_path=source_video_path, weight_path=weight_path)
    main(args)