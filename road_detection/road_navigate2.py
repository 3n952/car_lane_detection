import cv2
import numpy as np
import supervision as sv
import cv2
import numpy as np
import os
import warnings
# FutureWarning 무시
warnings.simplefilter(action='ignore', category=FutureWarning)

from tracking_direction import *

# 추적 객체의 진행 방향의 좌표를 입력하여 가우시안 필터를 적용하여 도로 영역을 추정
# 추정 과정을 시각화하는 모듈

def pred_road2pixel2(height,width, point_vector_list):
    '''
    차량의 진행방향을 기준으로 도로를 탐색하여 예측함
    args: frame (이미지 배열)
    returns: 도로로 예상되는 곳의 픽셀 위치를 반영한 이미지 (단일 채널)
    '''
    black_image = np.zeros((height,width,3), dtype=np.uint8)
    black_image2 = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    mask_image = black_image2.copy()

    for point in point_vector_list:
        point = tuple(map(int, point))
        cv2.circle(black_image2, point, radius=2, color=(255), thickness=-1)
    
    blurred_image = cv2.GaussianBlur(black_image2, (25, 25), sigmaX=4)

    # 도로 구분
    threshold = 20
    limit_pixel = int(height * 0.05)
    binary_image = np.where(blurred_image >= threshold, 255, 0).astype(np.uint8)
    mask_image[:-limit_pixel, :] = binary_image[:-limit_pixel,:]

    return mask_image

def main(args):

    #video에 대한 정보 - 높이 너비 fps 등등..
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    width = video_info.width
    height = video_info.height

    # yolov5 model load
    model = load_yolov5_model(args.weight_path)

    #tracking using supervision
    byte_track = sv.ByteTrack(
        frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold)

    thickness = sv.calculate_optimal_line_thickness(
        resolution_wh = video_info.resolution_wh)


    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    # 좌표 초기화 deque
    x_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))
    y_coordinates = defaultdict(lambda: deque(maxlen = video_info.fps))
    xyxy = defaultdict(list)
    
    # 방향 벡터 고려한 바운딩박스 접점 리스트
    points_vector = []

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
                    dx = x_coordinates_start - x_coordinates_end
                    dy = y_coordinates_start - y_coordinates_end
                    
                    x_min, y_min, x_max, y_max = xyxy[tracker_id][-1][0], xyxy[tracker_id][-1][1], xyxy[tracker_id][-1][2], xyxy[tracker_id][-1][3]
                    # x2_min, y2_min, x2_max, y2_max = xyxy[tracker_id][-5][0], xyxy[tracker_id][-5][1], xyxy[tracker_id][-5][2], xyxy[tracker_id][-5][3]
                    inter_point = find_boundary_intersection(x_min,y_min,x_max, y_max, dx, dy)
                    if inter_point != None:
                        inter_point = tuple(map(int, inter_point))
                        points_vector.append(inter_point)
                        cv2.circle(frame, inter_point,radius=4, color=(0,255,0), thickness=-1)
                    else: pass
                    # angle = math.degrees(math.atan2(dy, dx))  # 각도 계산
                    # labels.append(f"#{tracker_id} {int(angle)} degree")
                    ################################
            
            # 도로 영역 이미지 반환
            gray_scale_road = pred_road2pixel2(height, width, points_vector)
            # 도로 마스킹
            green_mask = np.zeros_like(frame)
            green_mask[:, :, 1] = gray_scale_road 
            blended_frame = cv2.addWeighted(frame, 1.0, green_mask, 0.3, 0)

            annotated_frame = blended_frame.copy()
            # box 시각화 클래스
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections)
        
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
    args = arguments(source_video_path=source_video_path,target_video_path=target_video_path, weight_path=weight_path)
    main(args)