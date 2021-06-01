import numpy as np
import cv2 as cv
import glob
import csv


def get_calibration_matrix():
    draw = True  # True when detected img pattern is drawn
    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9, 3), np.float32)
    objp[:,:2] = np.mgrid[0:6, 0:9].T.reshape(-1, 2)
    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    images = glob.glob('graphics/calibration_pattern/*.jpg')
    for fname in images:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (6, 9), None)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            if draw:
                # Draw and display the corners

                cv.drawChessboardCorners(img, (6, 9), corners, ret)
                img = cv.resize(img, (1152, 864))
                cv.imwrite('IMG2-detection.png', img)
                cv.imshow('img', img)
                cv.waitKey(0)
                cv.destroyAllWindows()

    img2 = cv.imread('graphics/calibration_pattern/snapshot_2021_03_22_15_27_51.jpg')
    gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    if True:
        dst = cv.undistort(img2, mtx, dist, None, None)
        cv.imwrite('graphics/calibration_output/calibresult.png', dst)

    with open('graphics/calibration_output/calibration_matrix.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow(mtx[0])
        writer.writerow(mtx[1])
        writer.writerow(mtx[2])
        writer.writerow(dist[0])

get_calibration_matrix()