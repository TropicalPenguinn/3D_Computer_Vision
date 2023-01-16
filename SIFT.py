import numpy as np
import cv2 

img1=cv2.imread('./study/Result_SIFT/sift_keypoints1.jpg')
img2=cv2.imread('./study/Result_SIFT/sift_keypoints2.jpg')

gray1=cv2.cvtColor(img1,cv2.COLOR_RGB2GRAY)
gray2=cv2.cvtColor(img2,cv2.COLOR_RGB2GRAY)

detector=cv2.xfeature2d.SIFT