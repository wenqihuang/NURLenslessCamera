import numpy as np
import scipy.fftpack as fftpack

n = 32
csrate = 0.3
m = int(0.5*n)
phi = (np.sign(np.random.rand(m,n)-0.5)+np.ones((m,n)))/2
# dct method
psi = np.zeros((n,n))
for i in range(n):
    ek = np.zeros((1,n))
    ek[0,i] = 1
    psi[:,i] = fftpack.idct(ek).T.reshape((n,))
    #print('phi',phi.shape,'psi',psi.shape)
theta = np.dot(phi,psi)

np.savetxt("theta_"+str(n)+"_"+str(csrate)+".txt", theta)

"""     theta_i_col = np.dot(phi,psi)
    theta.append(theta_i_col)
theta = np.array(theta).reshape((n,m)).T """