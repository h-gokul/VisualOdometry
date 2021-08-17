## Imports
import argparse

from misc.ReadCameraModel import ReadCameraModel
from misc.UndistortImage import UndistortImage
from misc.FundamentalMatrix import FundamentalMatrix, EssentialMatrix
from misc.utils import *
from misc.fileutils import *
from misc.PoseEstimation import * 
import matplotlib.pyplot as plt

    

##### main ######
BasePath = '../Data/'
K, LUT = CameraMatrix(BasePath)
frameSavepath = '../Outputs_UTpose/'
frameSavepath1 = '../Outputs_UTpose_cv2/'
foldercheck(frameSavepath)
foldercheck(frameSavepath1)

E_record = loadDict(name = 'Essential_1', )
print(len(E_record))
prev_pose = np.eye(4)
prev_pose1 = np.eye(4)

norm =  []
# for i in range(len(E_record)):
for i in range(1500):    
    print('frame ', str(i), '/', str(len(E_record)))
    
    E = np.array(E_record[str(i)])
    pts1_, pts2_ = loadSIFTmatches(i, siftFilepath = './data/sift_inliers/')
    
#     correctPose = recoverPose(pts1_, pts2_, E, optimize = False)

    retval, R, t, mask = cv2.recoverPose(E, pts1_, pts2_, K) # inbuilt pose estimation function
    correctPose1 = np.column_stack((R,t))    
    
#     print('norm:',np.linalg.norm(correctPose-correctPose1))
#     norm.append(np.linalg.norm(correctPose-correctPose1))
    
#     f.write(str(i) + '  norm: '+ str(np.linalg.norm(poses[best_i]- correctPose_ref)) + ',' + str(np.linalg.norm(pts3D- pts3D_ref))+ '\n')
        
    estimated_pose = np.vstack((correctPose1, np.array([0,0,0,1])))
    current_pose = prev_pose @ estimated_pose
    prev_pose = current_pose
    x_coordinate = current_pose[0, -1]
    z_coordinate = current_pose[2, -1]
    plt.scatter(x_coordinate, -z_coordinate, color='r')
    plt.savefig(frameSavepath1+str(i)+'.png', bbox_inches='tight')
    
#     estimated_pose1 = np.vstack((correctPose1, np.array([0,0,0,1])))
#     current_pose1 = prev_pose1 @ estimated_pose1
#     prev_pose1 = current_pose1
#     x_coordinate1 = current_pose1[0, -1]
#     z_coordinate1 = current_pose1[2, -1]
#     plt.scatter(x_coordinate1, -z_coordinate1, color='r')
#     plt.savefig(frameSavepath1+str(i)+'.png', bbox_inches='tight')

# np.save('norm.npy', np.array(norm))