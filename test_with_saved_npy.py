import numpy as np
from CSReconstruct import cs_reconstruct
import cv2


y = np.load("y_coarse.npy")
reimg = np.load("reimg.npy")
print(y)
#cv2.imshow("reimg",reimg)
#cv2.waitKey(0)


phi_coarse = np.load("phi_coarse.npy")
print(phi_coarse.shape)
height_coarse = 90
width_coarse = 120
cs_rate = 1
print(np.max(y))
reimg_low = cs_reconstruct(y.astype(float)*10,phi_coarse,height_coarse,width_coarse,cs_rate)
print(reimg_low)
reimg_low = (reimg_low - reimg_low.min())/(reimg_low.max()-reimg_low.min())*255
cv2.imshow("reimg",cv2.resize(reimg_low.astype(np.uint8),(160,120)))
cv2.waitKey(0)


