import cv2 
import random
import numpy as np
from numpy.linalg import svd
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
        

# 호모그래피 계산
iter = 500   # RANSAC 반복 횟수
e_limit = 100   # RANSAC error 임계값
num_list = list(range(len(good_matches)))

inlier_save = 0
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
  H = h.reshape(3,3)/s

  # inlier 수 계산
  inlier = 0
  for p in range(len(good_matches)):
    p1 = np.append(pts1[p], np.array([1]))
    p2 = H@p1
    p2 = p2/p2[2]
    error = np.linalg.norm(np.append(pts2[p], np.array([1])) - p2)
    if error < e_limit:
      inlier += 1

  # 최대 inlier 수를 갖는 호모그래피로 업데이트
  if H_save == []:
    H_save = H
  if inlier_save == 0:
    inlier_save = inlier
  if inlier > inlier_save:
    H_save = H
    inlier_save = inlier
	
# 후작업: inlier points로 최소제곱법
A2 = np.empty((0,9))
for n in inlier_save:
  A2 = np.append(A2,np.array([[pts1[n,0,0], pts1[n,0,1], 1, 0, 0, 0, -pts1[n,0,0]*pts2[n,0,0], -pts1[n,0,1]*pts2[n,0,0], -pts2[n,0,0]]]),axis=0)
  A2 = np.append(A2,np.array([[0, 0, 0, pts1[n,0,0], pts1[n,0,1], 1, -pts1[n,0,0]*pts2[n,0,1], -pts1[n,0,1]*pts2[n,0,1], -pts2[n,0,1]]]),axis=0)

# 특이값 분해
[U2,D2,Vh2] = np.linalg.svd(A2, full_matrices=True)

# 호모그래피 산출
h2 = Vh2[-1]
s2 = h2[-1]
H2 = h2.reshape(3,3)/s2

H_cv, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# OpenCV 함수로 계산한 
H_cv, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

# print('U',U.shape,': \n',U)
# print('D',D.shape,': \n',D)
# print('Vh',Vh.shape,': \n',Vh)
# print('A',A.shape,': \n',A)
# print('h: \n',h)
# print('s: \n',s)
print('H_save: \n',np.round(H_save))
print('H_cv: \n',np.round(H_cv))
print('inlier_save: \n',inlier_save)
