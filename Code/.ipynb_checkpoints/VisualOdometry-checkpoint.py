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
    Parser.add_argument('--frameSavepath', default='../UTPose_cv2/', help='Path to save Results, Default: ./Outputs/')
    
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
    E_record = loadDict(name= 'Essential_1') # empty dictionary returned
    
    for i in range(2919, len(impaths)-1):    
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
        saveSIFTmatches(i, pts1_, pts2_, siftFilepath = './data/sift_inliers/')
        
        E = EssentialMatrix(K, F)
        
        if str(i) not in E_record:
            E_record[str(i)] = E
            saveDict(E_record, 'Essential_1', Filepath = './data/record/')
        else:
            print(str(i), '### Already found in dictionary')
        print('frame ', str(i), '/', str(len(impaths)))

#         estimated_pose = recoverPose(pts1_, pts2_, E, optimize = False)
        
        retval, R, t, mask = cv2.recoverPose(E, pts1_, pts2_, K)
        estimated_pose = np.column_stack((R,t))    
        
        estimated_pose = np.vstack((estimated_pose, np.array([0,0,0,1])))
                    
        current_pose = prev_pose @ estimated_pose
        prev_pose = current_pose

        x_coordinate = current_pose[0, -1]
        z_coordinate = current_pose[2, -1]

        plt.scatter(x_coordinate, -z_coordinate, color='r')
#         trajectory_points.append(np.array([x_coordinate, z_coordinate]))
        print('\nframe_count', i)
        plt.savefig(frameSavepath+str(i)+'.png', bbox_inches='tight')

#     np.save('trajectory_points.npy', np.array(trajectory_points))

    
    
if __name__ == '__main__':
    main()

    