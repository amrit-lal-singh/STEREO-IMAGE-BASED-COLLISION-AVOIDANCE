import numpy as np
import cv2
from matplotlib import pyplot as plt

p_left = np.array([[640.0, 0.0, 640.0, 2176.0],
                   [0.0, 480.0, 480.0, 552.0],
                   [0.0, 0.0, 1.0, 1.4]])
p_right = np.array([[640.0, 0.0, 640.0, 2176.0],
                    [0.0, 480.0, 480.0, 792.0],
                    [0.0, 0.0, 1.0, 1.4]])

k_left, rotationL, T_vec_left, *_ = cv2.decomposeProjectionMatrix(p_left)
T_vec_left /= T_vec_left[3]
k_right, rotationR, T_vec_right, *_ = cv2.decomposeProjectionMatrix(p_right)
T_vec_right /= T_vec_right[3]

bike = cv2.imread("bike.png")[..., ::-1]

img_left = cv2.imread("left.png")[..., ::-1]
img_right = cv2.imread("right.png")[..., ::-1]
img_left_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
img_right_gray = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

matcher_type = 'sgbm'  # or 'sgbm'

if matcher_type == 'sgbm':
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=5 * 16,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=6,
        P1=8 * 3 * 6 ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * 6 ** 2,
        disp12MaxDiff=0,
        uniquenessRatio=0,
        speckleWindowSize=0,
        speckleRange=0,
        preFilterCap=0,
        mode=cv2.STEREO_SGBM_MODE_HH4
    )


# Computing the disparity map
disp_left = matcher.compute(img_left_gray, img_right_gray).astype(np.float32) / 16




f = k_left[0, 0]
b = T_vec_left[1, 0] - T_vec_right[1, 0]
disp_left[disp_left == 0] = 0.1
disp_left[disp_left == -1] = 0.1
depth_map_left = f * b / disp_left




cross_corr_map = cv2.matchTemplate(img_left, bike, method=cv2.TM_CCOEFF_NORMED)
obstacle_location = cv2.minMaxLoc(cross_corr_map)[3]


# Printing the bike distance


distance = depth_map_left[obstacle_location[0]+23,obstacle_location[1]+11 ]

print("The distance is ")
print(distance)
print("The location is ")
print(obstacle_location)

# Plotting the disparity map
plt.figure(figsize=(5,5))
plt.imshow(disp_left)
plt.show(block=False)
plt.pause(2)
plt.close()

# Plotting the depth map
plt.figure(figsize=(5, 5))
plt.imshow(depth_map_left, cmap='flag')
plt.show(block=False)
plt.pause(2)
plt.close()

# Display the cross correlation heatmap
plt.figure(figsize=(5, 5))
plt.imshow(cross_corr_map)
plt.show(block=False)
plt.pause(2)
plt.close()

