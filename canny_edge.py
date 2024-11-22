import numpy as np
import cv2
import warnings
warnings.filterwarnings('ignore')

import matplotlib as plt
import os
import supervision as sv

from road_detection.tracking_direction import arguments
from road_detection.road_navigate import *

'''
추출된 도로 영역에 대하여 canny edge 검출을 적용.
'''

def sobel_xy(img, orient='x', thresh=(20, 100)):
    """
    cv2.Sobel(src, ddepth, dx, dy, dst=None, ksize=None, scale=None,
    delta=None, borderType=None)
    --------------------------------------------------------------
    * parameters
    src : 입력 
    ddepth : 출력 데이터 타입(-1이면 입력과 동일한 데이터 타입)
    dx : x방향 미분의 차수
    dy : y방향 미분의 차수
    dst : 출력
    ksize : 커널 크기(default 3)
    scale : 연산 결과에 추가로 곱할 값(default 1)
    delta : 연산 결과에 추가로 더할 값(default 0)
    borderType : 가장자리 픽셀 확장 방식(default cv2.BORDER_DEFAULT)
    """
    
    if orient == 'x':
        # dx=1, dy=0이면 x 방향의 편미분
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))

    if orient == 'y':
        # dx=0, dy=1이면 y 방향의 편미분
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))

    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 255

    # Return the result
    return binary_output

def return_firstframe(args):
    # video에 대한 정보 - 높이, 너비, fps 등등..
    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    width = video_info.width
    height = video_info.height

    # 프레임 생성기
    frame_generator = sv.get_video_frames_generator(args.source_video_path)

    # 첫 번째 프레임 추출
    for frame in frame_generator:
        return height, width, frame  # 첫 번째 프레임 반환 후 함수 종료


if __name__ == '__main__':

    source_video_path = r'C:\Users\QBIC\Desktop\workspace\car_lane_detect\Yolov5\data\videos\test_video1.mp4'
    #target_video_path = r'C:\Users\QBIC\Desktop\workspace\car_lane_detect\Lane_detector\results\test_video.mp4'
    weight_path = r'C:\Users\QBIC\Desktop\workspace\car_lane_detect\Yolov5\weights\1280720_3rd.pt' 
    args = arguments(source_video_path=source_video_path, weight_path=weight_path)

    height, width, frame_img = return_firstframe(args)

    # road mask는 진행 방향 기준의 road detect
    road_mask = main(args)
    # red channel의 전체 픽셀
    temp = frame_img[:,:,2]
    # sobel filter X, Y
    #dx = cv2.Sobel(temp, -1, 1, 0)
    #dy = cv2.Sobel(temp, -1, 0, 1)
    dst = cv2.Canny(temp, 50, 150)
    
    # 이미지 크기 확인 (두 이미지가 동일한 크기라고 가정)
    if road_mask.shape != dst.shape:
        raise ValueError("Source Image and Road mask Image must have the same dimensions!")

    # 이미지 A에서 값이 255인 영역만 이미지 B에서 보이도록 마스크 적용
    result = np.zeros_like(dst)  # 검은 배경 생성
    result[road_mask == 255] = dst[road_mask == 255]  # 조건에 맞는 영역 복사

    # OpenCV로 시각화
    cv2.imshow('Canny edge detection in road area', result)
    cv2.waitKey(0)  # 키 입력 대기
    cv2.destroyAllWindows()  # 모든 창 닫기
        