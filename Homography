import cv2 
import pandas as pd
import numpy as np
from google.colab.patches import cv2_imshow

# 영상 불러오기
dir = '/content/drive/MyDrive/Colab Notebooks/OpenCV_test/'

img1 = cv2.imread(dir+'inha_logo.jpg',cv2.IMREAD_COLOR)
img2 = cv2.imread(dir+'inha_pitch',cv2.IMREAD_COLOR)
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

if img1 is None or img2 is None:
    print('Image load failed!')
    sys.exit()

# 특징점 알고리즘 객체 생성 (KAZE, AKAZE, ORB 등)
feature = cv2.KAZE_create() 
#feature = cv2.AKAZE_create()
#feature = cv2.ORB_create()

# 특징점 검출 및 기술자 계산
kp1, desc1 = feature.detectAndCompute(img1, None)
kp2, desc2 = feature.detectAndCompute(img2, None)

# 특징점 매칭
matcher = cv2.BFMatcher_create()
matches = matcher.match(desc1, desc2)

# 좋은 매칭 결과 선별
matches = sorted(matches, key=lambda x: x.distance)
good_matches = matches[:400]

print('# of kp1:', len(kp1))
print('# of kp2:', len(kp2))
print('# of matches:', len(matches))
print('# of good_matches:', len(good_matches))

# DMatch 객체에서 queryIdx와 trainIdx를 받아와서 크기와 타입 변환하기
pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches]
				).reshape(-1, 1, 2).astype(np.float32)    # 원본 keypoint
pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches]
				).reshape(-1, 1, 2).astype(np.float32)    # 이동 keypoint
        
from typing import Hashable
import random
import numpy as np
from numpy.linalg import svd

# 호모그래피 계산
iter = 300    # RANSAC 반복 횟수
num_list = list(range(len(good_matches)))

STE_save = 0
H_save = []
for i in range(iter):
  # 호모그래피 계산에 필요한 keypoint set 임의 선정 (1set은 4point로 구성)
  index_list = random.sample(num_list,4)
  A = np.empty((0,9))

  # DLT 계산 (matrix A 계산)
  for n in index_list:
    A = np.append(A,np.array([[pts1[n,0,0], pts1[n,0,1], 1, 0, 0, 0, -pts1[n,0,0]*pts2[n,0,0], -pts1[n,0,1]*pts2[n,0,0], -pts2[n,0,0]]]),axis=0)
    A = np.append(A,np.array([[0, 0, 0, pts1[n,0,0], pts1[n,0,1], 1, -pts1[n,0,0]*pts2[n,0,1], -pts1[n,0,1]*pts2[n,0,1], -pts2[n,0,1]]]),axis=0)

  # 특이값 분해
  [U,D,Vh] = np.linalg.svd(A, full_matrices=True)

  # 호모그래피 산출
  h = Vh[-1]
  s = h[-1]
  H = h.reshape(3,3)

  # STE(Symmetric Transfer Error) 계산
  STE = 0
  for p in range(len(good_matches)):
    p1 = np.append(pts1[0], np.array([1]))
    p2 = H@p1
    error = np.linalg.norm(np.append(pts2[0], np.array([1])) - p2)
    STE = STE + error
  
  # 최소 STE를 갖는 호모그래피로 업데이트
  if H_save == []:
    H_save = H
  if STE_save == 0:
    STE_save = STE
  if STE < STE_save:
    H_save = H
    STE_save = STE

H_cv, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

print('U',U.shape,': \n',U)
print('D',D.shape,': \n',D)
print('V',Vh.shape,': \n',Vh)
print('A',A.shape,': \n',A)
print('h: \n',h)
print('s: \n',s)
print('H_save: \n',H_save)
print('H_cv: \n',H_cv)
print('STE_save: \n',STE_save)


    
