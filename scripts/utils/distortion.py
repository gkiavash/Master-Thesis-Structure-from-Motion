import sys

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 8, 3), np.float32)
objp[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

# BASE_PATH = sys.argv[1]
BASE_PATH = '/home/gkiavash/Downloads/sfm_projects/datasets/calibration_3/images/*.jpg'
OUTPUT_PATH = "/home/gkiavash/Downloads/sfm_projects/datasets/calibration_3/out_cv.json"

images = glob.glob(BASE_PATH)
print(images)

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    print("finding for:", fname)
    ret, corners = cv.findChessboardCorners(gray, (8, 6), None)
    # print(ret, len(corners))
    print(ret)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (8, 6), corners2, ret)
        # cv.imshow('img', img)
        # cv.waitKey(500)
print()
# cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret", ret)
print("mtx", mtx)
print("dist", dist)
print("rvecs", rvecs)
print("tvecs", tvecs)

final_data = {
    "ret": ret,
    "mtx": mtx,
    "dist": dist,
    "rvecs": rvecs,
    "tvecs": tvecs,
}

import json

with open(OUTPUT_PATH, 'w') as f:
    json.dump(final_data, f)
