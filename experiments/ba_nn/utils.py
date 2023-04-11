import cv2
import numpy as np


def get_initial_matches_and_matrices():
    image_1 = "/content/drive/MyDrive/Master-Thesis-Structure-from-Motion/street_1/images/scene01961.jpg"
    image_2 = "/content/drive/MyDrive/Master-Thesis-Structure-from-Motion/street_1/images/scene01981.jpg"
    camera_params = [
        2288.0100739220425, 0.0, 960,
        0.0, 2284.529372201004, 545,
        0.0, 0.0, 1.0]
    fx, cx, fy, cy = camera_params[0], camera_params[2], camera_params[4], camera_params[5]

    # Load the two input images
    img1 = cv2.imread(image_1)
    img2 = cv2.imread(image_2)

    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and compute descriptors using SIFT
    sift = cv2.SIFT_create()
    # matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Match the keypoints using FLANN
    index_params = dict(algorithm=0, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Apply ratio test to select only good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Calculate the fundamental matrix
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    # Calculate the essential matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]]
    )  # Replace with camera intrinsic parameters
    E = np.matmul(np.matmul(np.transpose(K), F), K)

    # Decompose the essential matrix into rotation and translation matrices
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

    # Print the rotation and translation matrices
    print("Rotation Matrix:")
    print(R)
    print("Translation Vector:")
    print(t)

    ptsLeft = pts1[mask.ravel() == 1]
    ptsRight = pts2[mask.ravel() == 1]
    print("ptsLeft", len(ptsLeft))
    print("ptsRight", len(ptsRight))
    ptsLeft = np.reshape(ptsLeft, (ptsLeft.shape[0], ptsLeft.shape[2]))
    ptsRight = np.reshape(ptsRight, (ptsLeft.shape[0], ptsLeft.shape[2]))

    return K, R, t, F, E, ptsLeft, ptsRight



def rotation_matrix(angle_x, angle_y, angle_z):
    """
    Calculate a 3D rotation matrix for rotations around all three axes (x, y, z).
    Args are angle of rotation in radians.
    Returns np array: the rotation matrix.
    """
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])

    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])

    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])

    return np.dot(np.dot(Rz, Ry), Rx)


def calculate_keypoint_right_image(F, left_keypoint, right_image_shape):
    """
    Calculate the corresponding keypoint in the right image given the fundamental matrix
    and the corresponding keypoint in the left image.

    Parameters:
    - F (3x3 numpy array): the fundamental matrix
    - left_keypoint (2-element tuple): the keypoint in the left image (x, y)
    - right_image_shape (2-element tuple): the shape of the right image (height, width)

    Returns:
    - right_keypoint (2-element tuple): the keypoint in the right image (x, y)
    """

    # Homogeneous coordinates of the left keypoint
    x_l, y_l = left_keypoint
    left_homogeneous = np.array([x_l, y_l, 1])

    # Epipolar line in the right image
    right_epipolar_line = F @ left_homogeneous

    # Calculate the coordinates of the epipolar line intersection with the right image borders
    x_intercept_top = int(-right_epipolar_line[2] / right_epipolar_line[1])
    x_intercept_bottom = int(
        -(right_epipolar_line[2] + right_epipolar_line[0] * right_image_shape[0]) / right_epipolar_line[1])

    # Choose the intercept that is inside the image
    if (0 <= x_intercept_top < right_image_shape[1]):
        x_r = x_intercept_top
        y_r = 0
    elif (0 <= x_intercept_bottom < right_image_shape[1]):
        x_r = x_intercept_bottom
        y_r = right_image_shape[0]
    else:
        raise ValueError("The epipolar line doesn't intersect the right image.")

    right_keypoint = (x_r, y_r)

    return right_keypoint
