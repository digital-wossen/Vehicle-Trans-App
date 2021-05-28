import numpy as np
import cv2
import glob
import os
import matplotlib as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((9*12,3), np.float32)
objp[:,:2] = np.mgrid[0:12,0:9].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



# images = glob.glob('./Raw/Left_Lower/*.png')
images = glob.glob('./Raw/Left_Upper/*.png')

for fname in images:
    img = cv2.imread(fname)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    print("finding chessboard corners")
    ret, corners = cv2.findChessboardCorners(gray, (12,9),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        print("refining")
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (12,9), corners2,ret)
     
   
        
    # cv2.imwrite('Corners/Left_Lower/' + fname[len(fname)-28:len(fname)-4] + '_pattern.png',img)
    cv2.imwrite('Corners/Left_Upper/' + fname[len(fname)-28:len(fname)-4] + '_pattern.png',img)
    

    
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
# print(mtx)
# # f = open('calibration_matrix_lower', 'a')
# f = open('calibration_matrix_upper', 'a')
# f.write(str(mtx))
# f.close()





def undistort(img, mtx, dist, verbose=False):
    """
    Undistort the image givine the camera's distortion and matrix coefficients
    :param image: input image
    :param mtx: camera matrix
    :param dist: distortion coefficients
    """
    image_undistorted = cv2.undistort(
        img, mtx, dist, None, newCameraMatrix=mtx)

    if verbose:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(image_undistorted, cv2.COLOR_BGR2RGB))
        plt.show()

    return image_undistorted 

for fname in images:
    img = cv2.imread(fname)
    h,  w = img.shape[:2]

    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    # print(newcameramtx, roi)
    # undistort
    dst = undistort(img, mtx, dist)    

    # cv2.imwrite('Undistorted/Left_Lower/' + fname[len(fname)-28:len(fname)-4] + '_fixed.png',dst) 
    cv2.imwrite('Undistorted/Left_Upper/' + fname[len(fname)-28:len(fname)-4] + '_fixed.png',dst) 
