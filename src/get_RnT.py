import numpy.linalg as lg
from itertools import product

# Get rotation matrix and translation vector form a given homography matrix
# Written by JW_Choi

class get_RnT():
    def __init__(self, G, calib_mat):
        H_hat = lg.inv(calib_mat)@G@calib_mat
        _, s, _ = lg.svd(H_hat, full_matrices=True)
        self.H = H_hat/s[1]
        # self.H = matrix_G
        self.U, _, V = lg.svd(self.H, full_matrices=True)
        self.V = V.T
        self.v = np.concatenate((self.V[:,0:1][np.newaxis, :, :], self.V[:,2:3][np.newaxis, :, :]), axis=0)   #shape : (2,3,1)
        self.S = np.zeros((3,3))
        for i in range(3):
            self.S[i,i] = s[i]
        self.zetta = np.zeros(2)
        self.v_ = np.zeros((2,3,1)) 
        self.t_hat = np.zeros((2,3,1))
        self.t = np.zeros((3,4,1))
        self.n = np.zeros((2,3,1))
        self.R = np.zeros((4,3,3))

    def main_func(self):  # v_: v'  
        self.zetta[0] = (1/(2*self.S[0,0]*self.S[2,2]))*\
                        (-1 + np.sqrt(1 + 4*self.S[0,0]*self.S[2,2]/(self.S[0,0]-self.S[2,2])**2))
        self.zetta[1] = (1/(2*self.S[0,0]*self.S[2,2]))*\
                        (-1 - np.sqrt(1 + 4*self.S[0,0]*self.S[2,2]/(self.S[0,0]-self.S[2,2])**2))
        self.get_norm_v_()
        self.make_vecs()
        self.get_rot()
        
    def get_norm_v_(self):
        for j, i in enumerate(self.zetta):     
            norm_v_ = np.sqrt(i**2*(self.S[0,0] - self.S[2,2])**2 + 2*i*(self.S[0,0]*self.S[2,2] - 1) + 1)
            self.v_[j,:, :] = norm_v_ * self.v[j,:,:]

    def make_vecs(self):
        for i in range(2):
            self.t_hat[0,:,:] = (self.v_[0,:, :] + self.v_[1, : , :])/(self.zetta[0]- self.zetta[1])
            self.t_hat[1, :, :] =  - self.t_hat[0, : , :]
            self.n[0, :, :] = (self.zetta[0]*self.v_[1, : , :] + self.zetta[1]*self.v_[0, :, :])/\
                                (self.zetta[0]- self.zetta[1])
            self.n[1, : , :] = - self.n[0, : , :]
    def get_rot(self):
        idxes = np.array(np.meshgrid(np.array([0,1]), np.array( [0,1]))).T.reshape(-1,2)
        for i, j in enumerate(idxes):
            idx1, idx2 = j
            self.R[i, :, :] = np.matmul(self.H, lg.inv((np.eye(3) + np.matmul(self.t_hat[idx1, :, :], self.n[idx2, :, :].T))))
            self.t[:, i, :] = np.matmul(lg.inv(self.R[i, :, :].T), self.t_hat[idx1, :, :])
            
