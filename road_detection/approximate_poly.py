import cv2
import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from road_detector import *

'''
만들어진 도로영역에 대하여 보다 부드러운 도로 영역을 추출하기 위함. 
울퉁불퉁한 도로를 직선 혹은 곡선화 하는 과정
'''

source_video_path = r'C:\Users\QBIC\Desktop\workspace\car_lane_detect\Yolov5\data\videos\test_video1.mp4'
weight_path = r'C:\Users\QBIC\Desktop\workspace\car_lane_detect\Yolov5\weights\1280720_3rd.pt' 
args = arguments(source_video_path=source_video_path, weight_path=weight_path)

total_centeroid = main(args)

print('start approxiamting the road poly..')
print(total_centeroid)
input_data = np.array(total_centeroid)

# DBSCAN 모델 생성
dbscan = DBSCAN(eps=10, min_samples=30)
dbscan.fit(input_data)

labels = dbscan.labels_

# 수집된 중심점을 검은 이미지에 표시
black_image = np.zeros((580,720,3), dtype=np.uint8)

# 시각화
plt.figure(figsize=(8, 8))
plt.imshow(cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB))

# 데이터 포인트 시각화 (레이블에 따라 색상 변경)
unique_labels = set(labels)
colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))  # 다양한 색상 매핑

for label, color in zip(unique_labels, colors):
    if label == -1:
        # 노이즈는 흰색으로 표시
        cluster_color = 'white'
    else:
        cluster_color = color

    # 해당 클러스터에 속한 포인트만 시각화
    cluster_points = input_data[labels == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[cluster_color], label=f'Cluster {label}', s=10)

plt.title("DBSCAN Clustering on Black Background")
plt.legend()
plt.axis('off')  # 축 숨기기
plt.show()