## Imports
import argparse

from misc.ReadCameraModel import ReadCameraModel
from misc.UndistortImage import UndistortImage
from misc.FundamentalMatrix import FundamentalMatrix, EssentialMatrix
from misc.utils import *
from misc.fileutils import *
from misc.PoseEstimation import *
import matplotlib.pyplot as plt

def main():
    
    Parser = argparse.ArgumentParser()
    Parser.add_argument('--BasePath', default='../Data/', help='Data path of images, Default: ../Data/')
    
    Args = Parser.parse_args()
    BasePath = Args.BasePath
    
    
    K, LUT = CameraMatrix(BasePath)
    impaths =  ImagePaths(BasePath)[20:]
    print('TotalLength ', len(impaths))
    E_record = loadDict(name= 'EssentialMatrix') # empty dictionary returned
    print('prior EssentialMatrix data size: ',len(E_record))
    for i in range(len(impaths)):
        pts1_, pts2_ = loadSIFTmatches(i, siftFilepath = './data/sift_inliers/')
        print('shape: ',pts1_.shape, pts2_.shape)
        E = E_record(str(i))
        estimated_pose = recoverPose(pts1_, pts2_, E, optimize = False)
        
        estimated_pose = np.vstack((estimated_pose, np.array([0,0,0,1])))            
        current_pose = prev_pose @ estimated_pose
        prev_pose = current_pose

        x_coordinate = current_pose[0, -1]
        z_coordinate = current_pose[2, -1]
        plt.scatter(x_coordinate, -z_coordinate, color='r')
        plt.savefig(frameSavepath+str(i)+'.png', bbox_inches='tight')

        print('frame ', str(i), '/', str(len(impaths)))
        
    
if __name__ == '__main__':
    main()
