# -*- coding: utf-8 -*-

import cv2
import numpy as np
from CSReconstruct import cs_reconstruct
from PyQt5 import QtWidgets
from SignalCapture import LcdDisplayWindow, GetSensorData
import sys
import time 

if __name__ == "__main__":

    '''
    ### simulation old version
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
    roi_l = int(width_high * 0.33)
    roi_r = int(2.0*width_high * 0.33)
    roi_u = int(height_high * 0.33)
    roi_d = int(2.0*height_high * 0.33)
    roi_mask[roi_u:roi_d,roi_l:roi_r] = 1

    height_roi = roi_d - roi_u
    width_roi = roi_r - roi_l

    #cv2.imshow("roi_mask", roi_mask)
    #cv2.waitKey(0)

    

    cs_rate = 1
    n_high = height_high * width_high
    n_roi = height_roi * width_roi
    m_high = int(cs_rate * n_roi)

    print(height_roi,width_roi)
    


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
    '''
    ##################

    '''
    # simulation
    img = cv2.imread("img/sample_1024pixel.bmp",0)
    img = cv2.imread("img/test.jpeg",0)
    img = cv2.imread("img/timg2.jpeg",0)
    #img = cv2.imread("/Users/huangwenqi/Pictures/IMGP0726-1.jpg",0)

    cs_rate = 1
    (height_coarse, width_coarse) = (60, 80)
    (height_fine, width_fine) = (180, 240)
    img_hi = cv2.resize(img, (width_fine, height_fine))
    img_lo = cv2.resize(img, (width_coarse, height_coarse))
    print(img_hi.shape)
    '''

    cs_rate = 1
    (height_coarse, width_coarse) = (30, 40)
    (height_fine, width_fine) = (30, 40)
    #(x,y,x,y)
    roi_bbox = np.array((int(0.33*height_fine), int(0.33*width_fine),\
        int(0.66*height_fine), int(0.66*width_fine)))

    height_roi = roi_bbox[2] - roi_bbox[0]
    width_roi = roi_bbox[3] - roi_bbox[1]

    

    n_coarse_full = width_coarse * height_coarse
    n_fine_full = width_fine * height_fine
    n_fine_roi = width_roi * height_roi

    m_coarse = int(n_coarse_full * cs_rate)
    m_fine = int(n_fine_roi * cs_rate)

    phi_coarse = (np.random.rand(m_coarse, n_coarse_full) > 0.5).astype(int)
    phi_fine = (np.random.rand(m_fine, n_fine_roi) > 0.5).astype(int)

    masks_coarse = np.zeros((height_coarse,width_coarse,m_coarse))
    masks_fine = np.zeros((height_fine, width_fine,m_fine))

    #print(phi_coarse)
    print(phi_coarse[0,:])
    tmp = np.zeros((height_coarse,width_coarse))
    for i in range(m_coarse):
        tmp = phi_coarse[i,:].reshape((height_coarse,width_coarse))
        masks_coarse[:,:,i] = tmp
        

    tmp = np.zeros((height_roi,width_roi))
    for i in range(m_fine):
        tmp = phi_fine[i,:].reshape((height_roi,width_roi))
        masks_fine[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3],i] = tmp
        
    '''
    #sample simulation
    y_coarse = np.zeros((m_coarse,1))
    y_fine = np.zeros((m_fine,1))
    for i in range(m_coarse):
        y_coarse[i,0] = (masks_coarse[:,:,i] * img_lo).sum()
    
    for i in range(m_fine):
        y_fine[i,0] = (masks_fine[:,:,i] * img_hi).sum()
    '''

    # sample in real world
    app = QtWidgets.QApplication(sys.argv)
    a = LcdDisplayWindow()
    y_coarse = np.zeros((m_coarse,1))
    y_fine = np.zeros((m_fine,1))

    a.setimage(masks_coarse[:,:,0].reshape((height_coarse,width_coarse)))
    #cv2.imshow("mask",masks_coarse[:,:,0].reshape((height_coarse,width_coarse)))
    #cv2.waitKey(0)
    
    for i in range(m_coarse):
        a.setimage(masks_coarse[:,:,i].reshape((height_coarse,width_coarse)))
        time.sleep(1)
        y_coarse[i,0] = GetSensorData()
        print("y_coarse ", i, ": ", y_coarse[i,0])
    
    for i in range(m_fine):
        a.setimage(masks_fine[:,:,i].reshape((height_fine,width_fine)))
        time.sleep(1)
        y_fine[i,0] = GetSensorData()
        print("y_fine ", i, ": ", y_fine[i,0])

    np.save("./y_coarse.npy", y_coarse)
    np.save("./y_fine.npy", y_fine)

    print("Begin reconstruct ROI with high resolution.")
    reimg_roi = cs_reconstruct(y_fine,phi_fine,height_roi,width_roi,cs_rate)
    print("ROI reconstruction finished.")
    print("Begin reconstruct low-resolution image.")
    reimg_low = cs_reconstruct(y_coarse,phi_coarse,height_coarse,width_coarse,cs_rate)
    print("low-resolution reconstruction finished.")
    # notice that opencv has an inversed x-y axis
    reimg = cv2.resize(reimg_low, (width_fine,height_fine), interpolation=cv2.INTER_NEAREST)
    reimg[roi_bbox[0]:roi_bbox[2],roi_bbox[1]:roi_bbox[3]] = reimg_roi
    
    
    np.save('./reimg.npy',reimg)
    

    
    #cv2.imshow("origin image",img)

    cv2.imshow("NUR reconstruct image",reimg)
    cv2.waitKey(0)


