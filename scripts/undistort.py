import numpy as np
import cv2 as cv


def import_camera_params_from_opencv():
    mtx = np.array(
        [
            [2.30314339e+03, 0.00000000e+00, 2.56253024e+03],
            [0.00000000e+00, 2.31264462e+03, 1.92510736e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ]
    )
    dist = np.array(
        [
            [
                -0.25332645,
                0.09693553,
                -0.00118667,
                -0.00139854,
                -0.02359868
            ]
        ]
    )
    return mtx, dist


def import_camera_params_from_yaml():
    mtx = np.array([
        [2.2880100739220425e+03, 0.,                     2.5716169994772490e+03],
        [0.,                     2.2845293722010042e+03, 1.9205706165878657e+03],
        [0.,                     0.,                     1.]
    ])
    dist = np.array(
        [
            [
                -2.6434196998339271e-01,
                9.9571471417386093e-02,
                -2.4160314862664079e-04,
                -2.2267220647390027e-04,
                -1.9631169477800196e-02
            ]
        ]
    )
    return mtx, dist


IMAGE_PATH = "/home/gkiavash/Downloads/sfm_projects/datasets/calibration/G0040134.JPG"
IMAGE_PATH = "/home/gkiavash/Downloads/Master-Thesis-Structure-from-Motion/distorted_images/GOPR0037.JPG"
IMAGE_PATH = "/home/gkiavash/Downloads/sfm_projects/datasets/street_2/images/scene10141.jpg"


def undistort(IMAGE_PATH, mtx, dist, output_path, preview=True):

    img = cv.imread(IMAGE_PATH)
    h, w = img.shape[:2]
    cv.imshow('Undistorted Image', img)
    cv.waitKey(0)

    print(h, w)
    print(mtx)
    print(dist)
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), .5, (w, h))
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    print(newcameramtx)
    print(roi)
    if preview:
        cv.imshow('Undistorted Image', dst)
        cv.waitKey(0)

    x, y, w, h = roi
    dst = dst[y: y + h, x: x + w]

    cv.imwrite(output_path, dst)

    # cv.imshow('Undistorted Image', dst)
    # cv.imshow('distorted image', img)
    # cv.waitKey(0)
    cv.destroyAllWindows()


undistort(
    IMAGE_PATH,
    *import_camera_params_from_yaml(),
    # *import_camera_params_from_opencv(),
    output_path="/home/gkiavash/Desktop/undist.jpg"
)
