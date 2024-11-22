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
# 추적 객체의 bbox bottom_left, center, right의 좌표를 입력하여 가우시안 필터를 적용하여 도로 영역을 추정
# 추정 과정을 시각화하는 모듈

def pred_road2pixel(height,width,total_centeroid1=None,total_centeroid2=None,total_centeroid=None):

    '''
    도로를 예측할 수 있는 범위를 픽셀로 표현(gray scale 이미지 배열 반환)
    '''
    # 수집된 중심점을 검은 이미지에 표시
    black_image = np.zeros((height,width,3), dtype=np.uint8)
    gray_image = cv2.cvtColor(black_image, cv2.COLOR_BGR2GRAY)
    gray_image2 = gray_image.copy()

    if total_centeroid1 != None:
        for point in total_centeroid1:
            point = tuple(map(int, point))
            cv2.circle(gray_image, point, radius=2, color=(255), thickness=-1)  # 흰색 포인트

    if total_centeroid2 != None:
        for point in total_centeroid2:
            point = tuple(map(int, point))
            cv2.circle(gray_image, point, radius=2, color=(255), thickness=-1)
    
    if total_centeroid != None:
        for point in total_centeroid:
            point = tuple(map(int, point))
            cv2.circle(gray_image, point, radius=2, color=(255), thickness=-1)
        
    blurred_image = cv2.GaussianBlur(gray_image, (25, 25), sigmaX=4)

    # 도로 구분
    threshold = 50
    limit_pixel = int(height * 0.05)
    binary_image = np.where(blurred_image >= threshold, 255, 0).astype(np.uint8)
    gray_image2[:-limit_pixel, :] = binary_image[:-limit_pixel,:]

    return gray_image2

def main(args):
    #video에 대한 정보 - 높이 너비 fps 등등..
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    width = video_info.width
    height = video_info.height

    # yolov5 model load
    model = load_yolov5_model(args.weight_path)
    #video에 대한 정보 - 높이 너비 fps 등등..
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    width = video_info.width
    height = video_info.height
    # print(width, height)

    #tracking using supervision
    byte_track = sv.ByteTrack(frame_rate=video_info.fps, track_activation_threshold=args.confidence_threshold)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh = video_info.resolution_wh)
    bounding_box_annotator = sv.BoxAnnotator(thickness=thickness)
    
    # trace_annotator = sv.TraceAnnotator(
    #     thickness=thickness,
    #     trace_length=video_info.fps * 2,
    #     position=sv.Position.BOTTOM_CENTER)

    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    total_centeroid1 = []
    total_centeroid2 = []
    total_centeroid = []

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

            detections = byte_track.update_with_detections(detections= detections)

            left_car_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_LEFT)
            right_car_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_RIGHT)
            car_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER) 

            for car_list1, car_list2, car_list in zip(left_car_points, right_car_points, car_points):
                car_tuple1 = tuple(car_list1)
                car_tuple2 = tuple(car_list2)
                car_tuple = tuple(car_list)

                total_centeroid1.append(car_tuple1)
                total_centeroid2.append(car_tuple2)
                total_centeroid.append(car_tuple)
            
            # for center point
            #gray_scale_road = pred_road2pixel(height, width,total_centeroid= total_centeroid)

            # for bottom 양쪽 끝 point
            gray_scale_road = pred_road2pixel(height, width, total_centeroid1= total_centeroid1, total_centeroid2=total_centeroid2)

            green_mask = np.zeros_like(frame)
            green_mask[:, :, 1] = gray_scale_road 
            # Blend the green mask with the original frame
            blended_frame = cv2.addWeighted(frame, 1.0, green_mask, 0.3, 0)

            
            annotated_frame = blended_frame.copy()
            #annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # box 시각화 클래스
            annotated_frame = bounding_box_annotator.annotate(
                scene=annotated_frame, detections=detections)

            sink.write_frame(annotated_frame)
            cv2.imshow("annotated_frame", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
    cv2.destroyAllWindows()
    
    return gray_scale_road
    
if __name__ == '__main__':
   
    current_directory = os.getcwd()
    BASE = os.path.dirname(current_directory)
    source_video_path = BASE+r'\data\videos\test_video4.mp4'
    target_video_path = BASE+r'\results\road_detection\test_video4.mp4'

    weight_path = r'C:\Users\QBIC\Desktop\workspace\car_lane_detect\Yolov5\weights\1280720_3rd.pt' 
    args = arguments(source_video_path=source_video_path,target_video_path=target_video_path, weight_path=weight_path)
    main(args)