# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 00:12:36 2019

@author: ohman
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack



#编码矩阵生成
def make_encode(height,width):
    while True:
        #0~2范围随机矩阵(h^2,w^2)  (36,36)
        encoded = 2 * np.random.rand(height*width,height*width)

        #如果像素值大于阈值，则分配值（白色），否则分配另一个值（黑色）
        ret, encoded = cv2.threshold(encoded, 0.5 , 1 , cv2.THRESH_BINARY)
        
        if cv2.determinant(encoded) != 0:
            break
    return encoded

#root mean square error
def RMSE(img1,img2):
    n = len(img1)
    dif = img1 - img2
    dif2 = dif ** 2
    rmse = np.sqrt(np.sum(dif2) / (n))
    return rmse


def construct_haar_dwt_matrix(width_of_matrix):
    '''
    Create Haarwavelet transformation matrix H for the matrix vector
    mulplication implimentation of Haar wavelet transformation.
    This function uses the following nice formula to create the Haar
    transformation matrix:
                   H_n=1/sqrt(2)[H_(n/2) kron (1 1)
                                 I_(n/2) kron (1 -1)],
                                  where 'kron' denotes the kronecker product.
    The iteration starts with H_1=[1]. The normalization constant 1/sqrt(2)
    ensure that H_n^T*H_n=I, where I is identity matrix. Haar wavelets are the
    rows of H_n.
    '''
    level = np.log2(width_of_matrix)
    if 2**level<width_of_matrix:
        print('please ensure the value of input parameter is the power of 2')
    
    H=np.ones((1,1))
    NC=1/np.sqrt(2)
    LP=np.ones((1,2))
    HP=np.array([1,-1])

    for i in range(np.int8(level)):
        H=NC*np.row_stack((np.kron(H,LP),np.kron(np.eye(H.shape[0]),HP)))
        #print(H)

    return H

def main():

    img = cv2.imread("img/sample_256pixel.bmp",0)
    print("img shape" , img.shape)
    height, width = img.shape[:2]

    cs_rate = 1
    n = height*width
    m = int(cs_rate * n)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    
    #img_array (h,w)→(h*w,1) (36,1)
    img_array = img.reshape(height*width ,1)
    
    #编码矩阵Phi (h^2,w^2) (36,36)
    phi = (np.sign(np.random.rand(m,n)-0.5)+np.ones((m,n)))/2
    cv2.imshow("phi",phi)
    cv2.waitKey(0)

    y = np.dot(phi,img_array)

    psi = construct_haar_dwt_matrix(n) # wavelet transformation matrix
    #theta = np.dot(phi,psi) # wavelet
    theta = []
    for i in range(n):
        ek = np.zeros((1,n))
        ek[0,i] = 1
        psi = fftpack.idct(ek).T
        print('phi',phi.shape,'psi',psi.shape)
        theta_i_col = np.dot(phi,psi)
        theta.append(theta_i_col)

    theta = np.array(theta).reshape((n,m)).T
    
    
    #print(encoded.shape)
    #cv2.imshow("encoded", encoded)
    
    theta_inv = np.linalg.pinv(theta)
    alpha = np.dot(theta_inv,y)

    reconstruct = np.zeros((n,1))
    for i in range(n):
        ek = np.zeros((1,n))
        ek[0,i] = 1
        psi = fftpack.idct(ek).T
        reconstruct = reconstruct + np.dot(psi,alpha[i,0])

    #theta = np.array(theta).reshape((n,m)).T
    #reconstruct = np.dot(psi, alpha)
    reimg = reconstruct.reshape(height,width).astype("uint8")
    cv2.imshow("reconstruct img", reimg)
    cv2.waitKey(0)
        
"""     errors = []
    for i in range(height*width):
        tank = encoded[0:i+1,:]
        #print("tank",tank.shape) #(1,36)→(2,36),・・・(36,36)
        
        mask_inv = np.linalg.pinv(tank)
        #print("mask_inv",mask_inv.shape) #(36,1)→(36,2)・・・(36,36)
        
        output_array = np.dot(tank , img_array)
        #print("output_array",output_array.shape) #(i+1,36)・(36,1) = (i+1,1) 
        
        reconstruct = np.dot(mask_inv,output_array)
        #print("reconstruct",reconstruct.shape) #(36,i+1)・(i+1,1) = (36,1)
        
        reimg = reconstruct.reshape(height,width).astype("uint8")
        #print("reimg",reimg.shape) #(6,6)
        
        #cv2.imwrite("result/reconstruct_{}.bmp".format(i+1) , reimg)
        
        error = RMSE(img_array, reconstruct)
        errors.append(error)
    
    cv2.imshow("reconstruct img", reimg)
    cv2.waitKey(0)
    
    plt.figure(figsize=(12, 8)) 
    plt.ylabel("RMSE" , fontsize=25)
    plt.xlabel("iteration number", fontsize=25)
    plt.ylim(0,max(errors)+10)
    plt.tick_params(labelsize=20)
    plt.grid(which='major',color='black',linestyle='-')
    plt.plot(np.arange(1,65),errors)
    plt.show() """


if __name__ == "__main__":
    main()
"""     img = np.random.randn(8,1)
    H = construct_haar_dwt_matrix(8)



    c = np.dot(H,np.dot(img,H.T))
    rs = np.dot(np.dot(H.T,c),H)
    error = rs - img
    print(error) """




