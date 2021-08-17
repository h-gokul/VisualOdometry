import numpy as np

def recoverPose(pts1_, pts2_, E, optimize = False):
    
    """
    given a set of points and essential matrix, estimate pose
    """
    
    if (pts1_.shape[1] == 2) or (pts1_.shape[1] == 2): 
        pts1_,pts2_ = homo(pts1_),homo(pts2_)

    ### estimate all 4 mathematically possible poses###
    poses = estimatePoses(E)

    max_positiveZ, correctPose = 0, None
    
    ## choose the only physically possible pose
    for pose in poses:        
        ## get 3D points ##
        pts3D = LinearTriangulation(pts1_, pts2_, pose)  
        
        # if optimize:
            ## perform non linear optimization
            
        ### get n positive depths of the pose###
        n_positiveZ = ChieralityCheck(pts3D, pose)
        ## choose the pose with most positive depth results.
        if n_positiveZ >= max_positiveZ :
            max_positiveZ = n_positiveZ
            correctPose = pose

    return correctPose

def LinearTriangulation(pts1, pts2, pose):
    
    """
    To perform linear triangulation,
    np.cross(x,P)X = 0 and np.cross(x',P')X = 0 relationships need to be satisfied.
    X : 3D point 
    x : 2d image point 
    P : projection matrix

    Reference: http://www.cs.cmu.edu/~16385/s17/Slides/11.4_Triangulation.pdf
    """        
    # Following the general parameterization of dependant images, 
    #Pose1 is a  reference frame, Pose2 is our estimated pose
    Pose1, Pose2 = np.eye(3,4), pose
    pts3D = []
    for x1, x2 in zip(pts1, pts2):    
        x1P1 = cross_1Dx4D(x1,Pose1)
        x2P2 = cross_1Dx4D(x2,Pose2)
        A = np.vstack((x1P1, x2P2))

        _,_,Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X/X[-1]
        pts3D.append(X[:3]) 
    return np.array(pts3D)

def ChieralityCheck(pts3D, pose):
    r3 = pose[:,2]
    c = pose[:,2]
#     r3 = pose[2, :3]
#     c = pose[:, 3]
    n_positiveZ = 0
    for X in pts3D:
        # cheirality condition
        if (r3 @ (X - c)) > 0:
            n_positiveZ += 1
    return n_positiveZ

def homo(pts):
    return np.column_stack((pts, np.ones(len(pts)).reshape(-1,1)))

def cross_1Dx4D(x,P):
    """
    to find cross product between a 1D vector and 4D projection matrix
    """
    x_3x3 = np.array([[0, -x[2], x[1]],
                      [x[2], 0, -x[0]],
                      [-x[1], x[0], 0]])
    return x_3x3 @ P


def estimatePoses(E):
    U, _, Vt = np.linalg.svd(E, full_matrices=True)
    W = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]])

    R12 = U @ W @ Vt
    R34 = U @ W.T @ Vt
    C = U[:, 2]
    
    poses = np.array([np.column_stack((R12, C)),
                      np.column_stack((R12, -C)),
                      np.column_stack((R34, C)),
                      np.column_stack((R34, -C))])

    poses = signcheck(poses)
    
    return poses

def signcheck(poses):
    poses_ = []
    for pose  in poses:
        r = pose[:, :3]
        if np.linalg.det(r) <0:
            poses_.append(-pose)
        else:
            poses_.append(pose)
    return poses_
