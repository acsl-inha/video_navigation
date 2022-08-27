"""
This class finds homography matrix from given point matching pairs.

Calculating an optimal homography by using
  1. RANSAC algorithm
  2. LSM(Least Square Method)
  3. LSM with inlier points obtained from RANSAC

I lastly, compared the results with the OpenCV value.

Written by YJ_KIM (@ Muun-Muun)
"""


import random
import numpy as np
from numpy.linalg import svd

class find_homography:

  def cal_homography(self,point_index,pts1,pts2):
    # calculate DLT
    A = np.empty((0,9))
    for n in point_index:
      A = np.append(A,np.array([[pts1[n,0,0], pts1[n,0,1], 1, 0, 0, 0, -pts1[n,0,0]*pts2[n,0,0], -pts1[n,0,1]*pts2[n,0,0], -pts2[n,0,0]]]),axis=0)
      A = np.append(A,np.array([[0, 0, 0, pts1[n,0,0], pts1[n,0,1], 1, -pts1[n,0,0]*pts2[n,0,1], -pts1[n,0,1]*pts2[n,0,1], -pts2[n,0,1]]]),axis=0)

    # SVD(Singular Value Decomposition)
    [U,D,Vh] = np.linalg.svd(A, full_matrices=True)

    # calculate Homography
    """ 
    Vh의 마지막 행을 h로 사용
    Vh: V의 transpose 행렬
    h: 호모그래피 요소 벡터
    s: 호모그래피 scale 스칼라 값
    """
    h = Vh[-1]
    s = h[-1]
    H = h.reshape(3,3)/s

    return H

  def RANSAC(self,n_list,pts1,pts2,iter=500,e_limit=100):
    inlier_save = []
    H_save = []
    for i in range(iter):
      index_list = random.sample(n_list,4)
      H = self.cal_homography(index_list,pts1,pts2)

      # find inlier points
      inlier = []
      for p in n_list:
        p1 = np.append(pts1[p], np.array([1]))
        p2 = H@p1
        p2 = p2/p2[2]
        error = np.linalg.norm(np.append(pts2[p], np.array([1])) - p2)
        if error < e_limit:
          inlier.append(p)

      # update with homography with maximum number of inliers
      if H_save == []:
        H_save = H
      if inlier_save == 0:
        inlier_index_save = inlier
      if len(inlier) > len(inlier_save):
        H_save = H
        inlier_save = inlier

    return H_save, inlier_save

  def LSM(self,n_list,pts1,pts2):
    H = self.cal_homography(n_list,pts1,pts2)
    return H


if __name__ == '__main__':
  # OpenCV RANSAC
  H_cv, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
  
  # LSM
  H_lsm = find_homography.LSM(list(range(200)),pts1,pts2)

  # RANSAC
  H_ran, _ = find_homography.RANSAC(list(range(200)),pts1,pts2,300,100)

  # RANSAC + LSM
  H_temp,inlier = find_homography.RANSAC(list(range(200)),pts1,pts2,300,100)
  H_ran_lsm = find_homography.LSM(inlier,pts1,pts2)

  print('\n H using OpenCV:\n',H_cv)
  print('\n H using LSM:\n',H_lsm)
  print('\n H using RANSAC:\n',H_ran)
  print('\n H using RANSAC + LSM:\n',H_ran_lsm)
