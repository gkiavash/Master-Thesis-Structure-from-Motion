import numpy as np
import cv2 as cv

ret = 5.1014268271854295
mtx = np.array(
    [
        [2.30314339e+03, 0.00000000e+00, 2.56253024e+03],
        [0.00000000e+00, 2.31264462e+03, 1.92510736e+03],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ]
)
dist = np.array([[-0.25332645, 0.09693553, -0.00118667, -0.00139854, -0.02359868]])

IMAGE_PATH = "/home/gkiavash/Downloads/sfm_projects/datasets/calibration/G0040134.JPG"
IMAGE_PATH = "/home/gkiavash/Downloads/Master-Thesis-Structure-from-Motion/distorted_images/GOPR0037.JPG"
IMAGE_PATH = "/home/gkiavash/Downloads/sfm_projects/datasets/street_2/images/scene10141.jpg"

img = cv.imread(IMAGE_PATH)
h, w = img.shape[:2]
print(h, w)
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0.5, (w, h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)
print(roi)
cv.imshow('Undistorted Image', dst)
x, y, w, h = roi
dst = dst[y: y + h, x: x + w]

cv.imwrite("UNDIST_scene10141.jpg", dst)

# cv.imshow('Undistorted Image', dst)
# cv.imshow('distorted image', img)
# cv.waitKey(0)
cv.destroyAllWindows()
