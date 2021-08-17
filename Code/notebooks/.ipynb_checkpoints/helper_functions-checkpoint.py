# Imports
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import glob
import pprint
from scipy.optimize import least_squares

from misc.ReadCameraModel import ReadCameraModel
from misc.UndistortImage import UndistortImage
from random import sample

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

def findFeatures(im1, im2, ld = 0.7):
    
    # call ORB in OpenCV
    orb = cv2.ORB_create()
    # Compute ORB Keypoints
    im1_kpts, im1_descriptor = orb.detectAndCompute(gray(im1), None)
    im2_kpts, im2_descriptor = orb.detectAndCompute(gray(im2), None)
    
    if len(im1_kpts) == 0 or len(im2_kpts) == 0 :
        pts1,pts2 = [], []
        return pts1,pts2
    
    # im1_print,im2_print = im1.copy(),im2.copy()
    # cv2.drawKeypoints(im1, im1_kpts, im1_print, color = (255, 0, 0))
    # cv2.drawKeypoints(im2, im2_kpts, im2_print, color = (255, 0, 0))

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(np.float32(im1_descriptor), np.float32(im2_descriptor), 2)

    good_matches = []
    for m,n in matches:
        if m.distance < ld*n.distance:
            good_matches.append(m)

    pts1 = np.array([im1_kpts[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    pts2 = np.array([im1_kpts[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
    
    return pts1, pts2

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

def normalize(x):
    # compute the centroids
    meanx = np.mean(x[:,0])
    meany = np.mean(x[:,1])

# saw this in CMU site :(

#     s = np.sqrt(2)/(np.mean(np.sqrt((x[:,0]-meanx)**2 + (x[:,1] - meany)**2)))
#     Tn = np.array([[s,0,-meanx],[0,s,-meany],[0,0,1]]) 

    s = (np.mean(np.sqrt((x[:,0]-meanx)**2 + (x[:,1] - meany)**2)))
    Tn = np.array([[1/s,0,-meanx/s],[0,1/s,-meany/s],[0,0,1]])
    
    x = np.insert(x,2,1,axis=1)
    x = np.dot(Tn,x.T)
    x = x.T
    return x,Tn

def getFmatrix(x1,x2):
    
    x1,Tn1  = normalize(x1)
    x2,Tn2  = normalize(x2)

    n = x1.shape[0]
    #create correspondence matrix
    A = np.zeros((n,9))
    
    # Ai = [ui'ui, ui'vi, ui', vi'ui, vi'vi, vi', ui, vi, 1]
    for i in range(8):  
        A[i] = [x1[i,0]*x2[i,0], x1[i,0]*x2[i,1], x1[i,0], x1[i,1]*x2[i,0], x1[i,1]*x2[i,1], x1[i,1], x2[i,0],x2[i,1], 1]
    
    U, s, V = np.linalg.svd(A)

    F = V[:,8].reshape(3,3)
    U, s, V = np.linalg.svd(F)
    
    s[2] = 0
    F = np.dot(U,np.dot(np.diag(s),V))
    
    F = np.dot(Tn1.T,np.dot(F,Tn2))
    
    return F / F[2,2]

def errorCond(x1,x2,F):
    # To satisfy epipolar constraint
    x1 = x1.T
    err = np.matmul(x2,F)
    err = np.matmul(err,x1)
    err  =abs(np.squeeze(err))
    return err

def FmatrixRansac(pts1,pts2,M = 50,thresh = 0.01,in_built = False):
    
    if in_built:
        bestF,mask =  cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC, 1,0.90)
#         bestF,mask =  cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
        if bestF is None or len(bestF)==0:
            print('F bad :( ')
            return None, None
        U,s,V = np.linalg.svd(bestF)
        s[2] = 0
        bestF = np.dot(U,np.dot(np.diag(s),V))
        return bestF,mask
    else:
        # initialise:

        inliers, prev_inliers = list(), list()
        n_inliers, n_prev_inliers = len(prev_inliers), len(inliers)
        bestF = np.zeros((3,3))
        ind =  [i for i in range(len(pts1))]
        bestmask = np.zeros(len(ind))
        prev_mask = np.zeros(len(ind))
        mask = np.zeros(len(ind))
        # Run Ransac
        for i in range(M):
            inliers.clear()
            # sample random 8 points
            idx8 = sample(ind,8)

            X1 = pts1[idx8]
            X2 = pts2[idx8]

            # 8 point Fmatrix estimation
            F = getFmatrix(X1,X2)

            # For the given Fmatrix, find the inliers from all keypoints
            for j in ind:

                x1  = np.insert(pts1[j],2,1)
                x2  = np.insert(pts2[j],2,1)

                # compute error by epipolarity constraint
                error = errorCond(x1,x2,F)

                # if error below this degree
                if error<thresh : 
                    mask[j] = 1
#                     print(np.sum(mask))
                    # accept as inliers
                    inliers.append([x1.T,x2.T])

            if (len(inliers) > len(prev_inliers)):
                bestmask = mask
                bestF = F
                prev_mask = mask
                prev_inliers = inliers.copy()
                
    return bestF,bestmask

def getEssentialMatrix(K,F):
    E = K.T.dot(F).dot(K)
    U,s,V = np.linalg.svd(E)
    s = [1,1,0]
    E = np.dot(U,np.dot(np.diag(s),V))
    return E


def getCameraPose(E):
    ''' 
    To estimate camera pose from Essential matrix. 
    We get 4 types of poses out of this. 
    We need to find the best estimate of these poses
    
    '''
    W = np.array([[0,-1,0],[1,0,0],[0,0,-1]])
    U,S,V = np.linalg.svd(E)

    C1,R1 = U[:,2], U @ W @ V
    if int(np.linalg.det(R1))  == -1:
        C1,R1 = -C1,-R1

    C2,R2 = -U[:,2], U @ W @ V.T
    if int(np.linalg.det(R2))  == -1:
        C2,R2 = -C2,-R2

    C3,R3 = U[:,2], U @ W.T @ V.T
    if int(np.linalg.det(R1))  == -1:
        C3,R3 = -C3,-R3

    C4,R4 = -U[:,2], U @ W.T @ V.T
    if int(np.linalg.det(R1))  == -1:
        C4,R4 = -C4,-R4

    C1,C2,C3,C4 = C1.reshape(3,1),C2.reshape(3,1),C3.reshape(3,1),C4.reshape(3,1)

    k = 1
    P1 = np.concatenate((k*R1,C1),axis = 1)
    P2 = np.concatenate((k*R2,C2),axis = 1)
    P3 = np.concatenate((k*R3,C3),axis = 1)
    P4 = np.concatenate((k*R4,C4),axis = 1)

    return [P1,P2,P3,P4]


def getEulerAngles(R):
    import math
    eul = np.sqrt(R[0,0]**2 + R[1,0]**2 )
    singular_val  =  eul< 1e-6

    if singular_val:
        x = math.atan2(R[2,1],R[2,2])
        y = math.atan2(-R[2,0],eul)
        z = math.atan2(R[1,0],R[0,0])

    else:
        x = math.atan2(-R[1,2],R[1,1])
        y = math.atan2(-R[2,0],eul)
        z = 0    

    return x*180/math.pi, y*180/math.pi, z*180/math.pi

def LinearTriangulate(K,P0,P1,pt1,pt2):
    
    pt1 = np.insert(pt1,2,1)
    pt2 = np.insert(pt2,2,1)

    h_pt1 = np.matmul(np.linalg.inv(K),pt1)
    h_pt2 = np.matmul(np.linalg.inv(K),pt2)

    skew1 = skew(h_pt1)
    skew2 = skew(h_pt2)

    P0 = np.matmul(skew1,P0)
    P1 = np.matmul(skew1,P1)
    A = np.vstack([P0,P1])
    _,_,v = np.linalg.svd(A)
    pt3d = v[-1]
    pt3d = pt3d/pt3d[3]
    
    return pt3d[0:3].reshape(3,1)

def skew(v):
    a=v[0]
    b=v[1]
    c=v[2]
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

def poseEstimation(K,E,pts1,pts2):
    poses = getCameraPose(E)
    
    H0  = np.identity(4)
    P0 = H0[:3,:]
    prev = 0
    bestT = np.zeros((3,1))
    bestR = np.zeros((3,3))

    for p,P1 in enumerate(poses):
        R = P1[:,:3]
        T = P1[:,3].reshape(3,1)
        count = 0
    #     x,y,z = getEulerAngles(R)
    #     print(x,y,z)

        # ignpre if the pose is too tilted
    #     if x>-50 and x<50  and y<-50 and y>50 :
    #         print('Pose',i)
    #         find how many of the estimated 3d points satisfy the cheirality condition
        for i in range(len(pts1)):

            pts3d = LinearTriangulate(K,P0,P1,pts1[i],pts2[i])
            cond = np.matmul(R[2,:],(pts3d - T)) 

            if cond > 0: 
                count += 1
    #     The pose with maximum no. of points satisfying cheirality condition is our best choice
        if count > prev:
            prev = count
            if T[2]>0:
                T = -T
            bestT = T
            bestR = R

    pose = np.hstack((bestR,bestT))
    return pose
########Sanch method--------------------------------------------------------------------------------------------------------------------------------------

def extract_camera_pose(E):
    poses = []
    W = np.array(([0,-1,0],[1,0,0],[0,0,1]))
    U,S,V = np.linalg.svd(E)
    
    C1 = -U[:,2].reshape(-1,1)
    C2 = U[:,2].reshape(-1,1)
    C3 = -U[:,2].reshape(-1,1)
    C4 =  U[:,2].reshape(-1,1)


    R1 = np.matmul(np.matmul(U,W),V)
    R2 = np.matmul(np.matmul(U,W),V)
    R3 = np.matmul(np.matmul(U,W.T),V)
    R4 = np.matmul(np.matmul(U,W.T),V)

    if np.linalg.det(R1)<0:
        R1=-R1
    
    if np.linalg.det(R2)<0:
        R2=-R2
    
    if np.linalg.det(R3)<0:
        R3=-R3
    
    if np.linalg.det(R4)<0:
        R4=-R4
    
    P1 = np.concatenate((R1,C1),axis = 1)
    poses.append(P1)
    P2 = np.concatenate((R2,C2),axis = 1)
    poses.append(P2)
    P3 = np.concatenate((R3,C3),axis = 1)
    poses.append(P3)
    P4 = np.concatenate((R4,C4),axis = 1)
    poses.append(P4)

    return poses

def skew(v):
    a=v[0]
    b=v[1]
    c=v[2]
    return np.array([[0,-c,b],[c,0,-a],[-b,a,0]])

def linear_triangulation(K,P0,P1,pt1,pt2):
    pt1 = np.insert(np.float32(pt1),2,1)
    pt2 = np.insert(np.float32(pt2),2,1)
    # print(pt1.shape)
    homo_pt1 = np.matmul(np.linalg.inv(K),pt1.reshape((-1,1)))
    homo_pt2 = np.matmul(np.linalg.inv(K),pt2.reshape((-1,1)))

    skew0 = skew(homo_pt1.reshape((-1,)))
    skew1 = skew(homo_pt2.reshape((-1,)))

    P0 = np.concatenate((P0[:,:3], -np.matmul(P0[:,:3],P0[:,3].reshape(-1,1))),axis=1)
    P1 = np.concatenate((P1[:,:3], -np.matmul(P1[:,:3],P1[:,3].reshape(-1,1))),axis=1)
    # P0 = homogeneousMat(P0)
    # P1 = homogeneousMat(P1)
    pose1 = np.matmul(skew0,P0[:3,:])
    pose2 = np.matmul(skew1,P1[:3,:])

    #Solve the equation Ax=0
    A = np.concatenate((pose1,pose2),axis=0)
    u,s,vt = np.linalg.svd(A)
    X = vt[-1]
    X = X/X[3]
    return X

def find_correct_pose(P0,poses, allPts):
    max = 0
    flag = False
    for i in range(4):
        P = poses[i]
        # print("Each"+str(i),P)
        r3 = P[:,3]
        r3 = np.reshape(r3,(1,3))
        C = P[:,3]
        C = np.reshape(C,(3,1))
        pts_list = allPts[i]
        pts = np.array(pts_list)
        pts = pts[:,0:3].T

        diff = np.subtract(pts,C)
        Z = np.matmul(r3,diff)
        Z = Z>0
        _,idx = np.where(Z==True)
        # print(idx.shape[0])
        if max < idx.shape[0]:
            poseid = i
            correctPose = P
            indices = idx
            max = idx.shape[0]
    if max==0:
        flag = True
        correctPose = None
    return correctPose,flag,poseid
