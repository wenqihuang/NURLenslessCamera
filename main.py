# -*- coding: utf-8 -*-

import cv2
import numpy as np
from CSReconstruct import cs_reconstruct

if __name__ == "__main__":
    img = cv2.imread("img/sample_1024pixel.bmp",0)
    img = cv2.imread("img/test.jpeg",0)
    img = cv2.imread("img/timg2.jpeg",0)

    # prepare high/low res image
    img_highres = cv2.resize(img,(256,256))
    img_lowres = cv2.resize(img,(64,64))

    height_high, width_high = img_highres.shape[:2]
    height_low, width_low = img_lowres.shape[:2]

    # prepare roi mask
    roi_mask = np.zeros((height_high,width_high))
    roi_l = int(width_high/3.0)
    roi_r = int(2.0*width_high/3.0)
    roi_u = int(height_high/3.0)
    roi_d = int(2.0*height_high/3.0)
    roi_mask[roi_u:roi_d,roi_l:roi_r] = 1

    height_roi = roi_d - roi_u
    width_roi = roi_r - roi_l

    #cv2.imshow("roi_mask", roi_mask)
    #cv2.waitKey(0)

    

    cs_rate = 0.6
    n_high = height_high*width_high
    n_roi = height_roi * width_roi
    m_high = int(cs_rate * n_roi)

    n_low = height_low * width_low
    m_low = int(cs_rate * n_low)

    

    #cv2.imshow("img", img)
    #cv2.waitKey(0)


    #img_array (h,w)→(h*w,1) (36,1)
    img_array_highres = img_highres.reshape(n_high,1)
    img_array_lowres = img_lowres.reshape(n_low,1)

    roi_mask_array = roi_mask.reshape(n_high,1)



    #编码矩阵Phi (h^2,w^2) (36,36)
    phi_low = (np.sign(np.random.rand(m_low,n_low)-0.5)+np.ones((m_low,n_low)))/2
    y_low = np.dot(phi_low, img_array_lowres)
    
    phi_high = (np.sign(np.random.rand(m_high,n_high)-0.5)+np.ones((m_high,n_high)))/2
    phi_high = phi_high * roi_mask_array.T
    #print(np.max(phi_high))
    #cv2.imwrite("phi_high.jpg", phi_high*255)
    #cv2.waitKey(0)

    y_high = np.dot(phi_high, img_array_highres)

    # high res with black edge
    #cs_reconstruct(y_high,phi_high,height_high,width_high,cs_rate)


    phi_roi = np.zeros((m_high,n_roi))
    col_idx = 0
    for i in range(n_high):
        if roi_mask_array[i,0] == 1:
            phi_roi[:,col_idx] = phi_high[:,i]
            col_idx += 1

    
    #print(np.max(phi_roi))
    #cv2.imwrite("phi_roi.jpg", phi_roi*255)
    #cv2.waitKey(0)

    print("Begin reconstruct ROI with high resolution.")
    reimg_roi = cs_reconstruct(y_high,phi_roi,height_roi,width_roi,cs_rate)
    print("ROI reconstruction finished.")
    print("Begin reconstruct low-resolution image.")
    reimg_low = cs_reconstruct(y_low,phi_low,height_low,width_low,cs_rate)
    print("low-resolution reconstruction finished.")
    reimg = cv2.resize(reimg_low, (height_high,width_high), interpolation=cv2.INTER_NEAREST)
    reimg[roi_u:roi_d,roi_l:roi_r] = reimg_roi

    
    cv2.imshow("origin image",img)

    cv2.imshow("NUR reconstruct image",reimg)
    cv2.waitKey(0)


