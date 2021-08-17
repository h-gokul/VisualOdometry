# getCameraMatrix
# Imports
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pprint
from scipy.optimize import least_squares

from misc.ReadCameraModel import ReadCameraModel
from misc.UndistortImage import UndistortImage
from random import sample
from misc.helper_functions import *

root = '../Data/model'
fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel(root)
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

print(K)

## load image list
DataPath = '../Data/stereo/centre/'

impaths = []
for f in sorted(glob.glob(DataPath+"*.png")):
    impaths.append(f)

print(len(DataPath))
H0  = np.identity(4)
p_0 = np.array([[0, 0, 0, 1]]).T
flag = 0

data_points = []
prev_pts1, prev_pts2 = None, None
prev_F, prev_mask = None, None

for i in range(20,len(impaths)-1):
    print('Frame: ', i)

    im1 = UndistortImage(cv2.cvtColor(cv2.imread(impaths[i],0), cv2.COLOR_BAYER_GR2RGB),LUT)
    im2 = UndistortImage(cv2.cvtColor(cv2.imread(impaths[i+1],0), cv2.COLOR_BAYER_GR2RGB),LUT)

    pts1,pts2 = findKeypoints(im1,im2)
#     pts1,pts2 = findFeatures(im1,im2)
    
    if len(pts2) == 0 or len(pts1) == 0:
        print('none returned - level1')
        pts1, pts2 = prev_pts1, prev_pts2
        F, mask = prev_F, prev_mask  
    else:
        F,mask  = FmatrixRansac(pts1,pts2,in_built = True) # False - selfmade Ransac code; True-  inbuilt cv2 function
        if F is not None:
            prev_pts1, prev_pts2 = pts1, pts2
            prev_F, prev_mask = F, mask
        else:
            pts1, pts2 = prev_pts1, prev_pts2
            F, mask = prev_F, prev_mask
    
    E = getEssentialMatrix(K,F)
    
#         # only inliers
    print('mask: ',len(mask) ,'pts: ', len(pts2))
    pts1_ = pts1[mask.ravel() == 1]
    pts2_ = pts2[mask.ravel() == 1]
    
    retval, R, t, mask = cv2.recoverPose(E, pts1_, pts2_, K) # inbuilt pose estimation function
    pose = np.column_stack((R,t))    
    
    #     pose = poseEstimation(K,E,pts1,pts2) # pose estimation from scratch
    
    a = np.array([0, 0, 0, 1])
    H = np.vstack((pose, a))

    H0 = H0 @ H
    p_projection = H0 @ p_0

    print('x- ', p_projection[0],'\t','y- ', p_projection[2])
    
    data_points.append([p_projection[0][0], -p_projection[2][0]])
    plt.scatter(p_projection[0][0], -p_projection[2][0], color='r')
    
    if cv2.waitKey(0) == 27:
        break
    flag = flag + 1
    cv2.destroyAllWindows()
plt.savefig('G_Vo_Finb_Pinb.png')
plt.show()
np.save('data_pts.npy', np.array(data_points))

# data_points.to_csv('G_Vo_Finb_Pinb.csv')
    
    # sanket OpenCV pose estimation script
    
    ## Sanchit pose estimation script
#     poses = extract_camera_pose(E)
#     cumulative_points = dict()

#     for j in range(4):
#         X = []
#         for i in range(len(pts1)):
#             pt = linear_triangulation(K,P0,poses[j],pts1[i],pts2[i])
#             X.append(pt)
#                # print("Pose" + str(j))
#         cumulative_points.update({j:X})

#     correctPose,no_inlier,poseid = find_correct_pose(P0, poses, cumulative_points)
#     R_custom = correctPose[:,:3].reshape(3,3)
#     C_custom = correctPose[:,3].reshape(3,1)

#     if np.linalg.det(R_custom)<0:
#         R_custom = -R_custom

#     H2_custom = np.hstack((R_custom,np.matmul(-R_custom,C_custom)))
#     H2_custom = np.vstack((H2_custom,[0,0,0,1]))

#     H1_custom = np.matmul(H1_custom,H2_custom)
#     pos_custom = np.matmul(-H1_custom[:3,:3].T,H1_custom[:3,3].reshape(-1,1))

#     print("Frame number: ",i)

#     x_camera = H1_custom[0,3]

#     z_camera = H1_custom[2,3]

#     plt.plot(x_camera,-z_camera,'.r')

#     plt.pause(0.01)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
        
# import pandas as pd
# df = pd.DataFrame(data_points, columns = ['X', 'Y'])
# df.to_excel('VOmap_1.xlsx')
