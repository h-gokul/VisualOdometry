
from misc.ReadCameraModel import ReadCameraModel
from misc.UndistortImage import UndistortImage
import cv2
import numpy as np
import glob


def CameraMatrix(BasePath):
    path = BasePath + 'model'
    a, b, cx, cy, G_camera_image, LUT = ReadCameraModel(path)
    K = np.array([[a , 0 , cx],[0 , b , cy],[0 , 0 , 1]]) # Kamera Matrix        
    
    return K, LUT
                           
def ImagePaths(BasePath):    
    DataPath = BasePath + 'stereo/centre/'
    impaths = []
    for f in sorted(glob.glob(DataPath+"*.png")):
        impaths.append(f)
    return impaths

def preprocess(impaths,LUT, i):
    im1 = UndistortImage(bayer2rgb(cv2.imread(impaths[i],0)), LUT)[:800,:,:]
    im2 = UndistortImage(bayer2rgb(cv2.imread(impaths[i+1],0)), LUT)[:800,:,:]
    return im1,im2

def bayer2rgb(im):
    return cv2.cvtColor(im, cv2.COLOR_BAYER_GR2RGB)

def gray(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)

def drawlines(im1,im2,lines,pts1,pts2):
    '''
    reference: https://answers.opencv.org/question/38682/strange-behavior-of-findfundamentalmat-ransac/
    
    img1 - image on which we draw the epilines for the points in img2
    lines - corresponding epilines
    '''
    lines = lines.reshape(-1,3)
    img1 = im1.copy()
    img2 = im2.copy()

    r,c = img1.shape[:2]
#     img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
#     img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(100,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(np.int32(pt1)),10,color,-1)
        img2 = cv2.circle(img2,tuple(np.int32(pt2)),10,color,-1)
    return img1,img2


def SIFTpoints(im1,im2):
    MIN_MATCH_COUNT = 10
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im1,None)
    kp2, des2 = sift.detectAndCompute(im2,None)


    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    pts1 = np.array([kp1[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts2 = np.array([kp2[m.trainIdx].pt for m in good]).reshape(-1, 2)

    return pts1,pts2