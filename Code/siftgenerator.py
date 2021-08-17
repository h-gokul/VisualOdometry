## Imports
import argparse

from misc.ReadCameraModel import ReadCameraModel
from misc.UndistortImage import UndistortImage
from misc.FundamentalMatrix import FundamentalMatrix, EssentialMatrix
from misc.utils import *
from misc.fileutils import *
from misc.PoseEstimation import *
import matplotlib.pyplot as plt

## 
def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../Data/', help='Data path of images, Default: ../Data/')
    Parser.add_argument('--frameSavepath', default='./Outputs_cv2Pose/', help='Path to save Results, Default: ./Outputs/')
    
    Args = Parser.parse_args()
    BasePath = Args.BasePath
    frameSavepath = Args.frameSavepath
    
    foldercheck(frameSavepath) 
    load_sift =False
    trajectory_points = []   
    prev_pose = np.eye(4)
#     pose_record = loadPoses() # empty dict initially
    
    K, LUT = CameraMatrix(BasePath)
    impaths =  ImagePaths(BasePath)[20:]
    E_record = {}
    
    for i in range(len(impaths)):
        im1, im2 = preprocess(impaths,LUT, i)
        if load_sift == False:
            pts1,pts2 = SIFTpoints(im1, im2)
            if len(pts1)>0 or len(pts2) >0:
                saveSIFTmatches(i, pts1, pts2)
            else:
                print('COULDNOT SAVE SIFT POINT at :', i)
        else:
            pts1, pts2 = loadSIFTmatches(i, siftFilepath = './data/sift/')
        
        data = (pts1,pts2)
        F,inlier_mask = FundamentalMatrix(data ,s = 8, thresh = 0.0001,n_iterations = 75)
        pts1_ = pts1[inlier_mask==1]
        pts2_ = pts2[inlier_mask==1]    

        E = EssentialMatrix(K, F)
        E_record[str(i)] = E
        saveDict(E_record, 'EssentialMatrix_1', Filepath = './data/record/')
        print('\nframe ', i, ' Done')
        
#         estimated_pose = recoverPose(pts1_, pts2_, E, optimize = False)
        
        retval, R, t, mask = cv2.recoverPose(E, pts1_, pts2_, K) # inbuilt pose estimation function
        estimated_pose = np.column_stack((R,t))    

        estimated_pose = np.vstack((estimated_pose, np.array([0,0,0,1])))
        
        if str(i) not in pose_record:
            pose_record[str(i)] =  estimated_pose
        else:
            print('pose found, not being recorded in json')
            
        current_pose = prev_pose @ estimated_pose
        prev_pose = current_pose

        x_coordinate = current_pose[0, -1]
        z_coordinate = current_pose[2, -1]

        plt.scatter(x_coordinate, -z_coordinate, color='r')
        trajectory_points.append(np.array([x_coordinate, z_coordinate]))
        print('\nframe_count', i)
        plt.savefig(frameSavepath+str(i)+'.png', bbox_inches='tight')

#     savePoses(pose_record)
    np.save('cv2Pose_trajectory_points.npy', np.array(trajectory_points))

    
    
if __name__ == '__main__':
    main()

    