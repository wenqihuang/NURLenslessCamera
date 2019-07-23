import numpy as np
import scipy.fftpack as fftpack

n_list = [32**2, 64**2, 128**2]
csrate_list = [0.2,0.4,0.6,0.8,1.0]
for ni in range(len(n_list)):
    for j in range(len(csrate_list)):
        print(ni,j)
        n = n_list[ni]
        csrate = csrate_list[j]
        m = int(csrate*n)
        phi = (np.sign(np.random.rand(m,n)-0.5)+np.ones((m,n)))/2
        # dct method
        psi = np.zeros((n,n))
        for i in range(n):
            ek = np.zeros((1,n))
            ek[0,i] = 1
            psi[:,i] = fftpack.idct(ek).T.reshape((n,))
            #print('phi',phi.shape,'psi',psi.shape)
        #theta = np.dot(phi,psi)

        np.save("./pre_calculation/psi_"+str(np.int(np.sqrt(n)))+"_"+str(csrate)+".npy", psi)

"""     theta_i_col = np.dot(phi,psi)
    theta.append(theta_i_col)
theta = np.array(theta).reshape((n,m)).T """