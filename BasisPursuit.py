# -*- coding: utf-8 -*-


from cvxopt import matrix, mul, div, cos, sin, exp, sqrt 
from cvxopt import blas, lapack, solvers
from numpy import array
import pylab


def basis_pursuit(A,y):
    # Basis pursuit problem
    #
    #     minimize    ||A*x - y||_2^2 + ||x||_1
    #
    #     minimize    x'*A'*A*x - 2.0*y'*A*x + 1'*u
    #     subject to  -u <= x <= u
    #
    # Variables x (n),  u (n).

    # convert np array to matrix
    A = matrix(A)
    y = matrix(y)

    #print(A.size)
    #print(y.size)

    m, n = A.size
    r = matrix(0.0, (m,1))

    q = matrix(1.0, (2*n,1))
    blas.gemv(A, y, q, alpha = -2.0, trans = 'T')


    def P(u, v, alpha = 1.0, beta = 0.0):
        """
        Function and gradient evaluation of

        v := alpha * 2*A'*A * u + beta * v
        """

        blas.gemv(A, u, r)      
        blas.gemv(A, r, v, alpha = 2.0*alpha, beta = beta, trans = 'T') 
            

    def G(u, v, alpha = 1.0, beta = 0.0, trans = 'N'):
        """
        v := alpha*[I, -I; -I, -I] * u + beta * v  (trans = 'N' or 'T')
        """

        blas.scal(beta, v) 
        blas.axpy(u, v, n = n, alpha = alpha) 
        blas.axpy(u, v, n = n, alpha = -alpha, offsetx = n) 
        blas.axpy(u, v, n = n, alpha = -alpha, offsety = n) 
        blas.axpy(u, v, n = n, alpha = -alpha, offsetx = n, offsety = n) 


    h = matrix(0.0, (2*n,1))


    # Customized solver for the KKT system 
    #
    #     [  2.0*A'*A   0    I      -I     ] [x[:n] ]     [bx[:n] ]
    #     [  0          0   -I      -I     ] [x[n:] ]  =  [bx[n:] ].
    #     [  I         -I   -D1^-1   0     ] [z[:n] ]     [bz[:n] ]
    #     [ -I         -I    0      -D2^-1 ] [z[n:] ]     [bz[n:] ]
    #
    # where D1 = W['di'][:n]**2,  D2 = W['di'][:n]**2.
    #    
    # We first eliminate z and x[n:]:
    #
    #     ( 2*A'*A + 4*D1*D2*(D1+D2)^-1 ) * x[:n] = 
    #         bx[:n] - (D2-D1)*(D1+D2)^-1 * bx[n:] 
    #         + D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bz[:n]
    #         - D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bz[n:]           
    #
    #     x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bz[:n]  - D2*bz[n:] ) 
    #              - (D2-D1)*(D1+D2)^-1 * x[:n]         
    #
    #     z[:n] = D1 * ( x[:n] - x[n:] - bz[:n] )
    #     z[n:] = D2 * (-x[:n] - x[n:] - bz[n:] ).
    #
    #
    # The first equation has the form
    #
    #     (A'*A + D)*x[:n]  =  rhs
    #
    # and is equivalent to
    #
    #     [ D    A' ] [ x:n] ]  = [ rhs ]
    #     [ A   -I  ] [ v    ]    [ 0   ].
    #
    # It can be solved as 
    #
    #     ( A*D^-1*A' + I ) * v = A * D^-1 * rhs
    #     x[:n] = D^-1 * ( rhs - A'*v ).

    S = matrix(0.0, (m,m))
    Asc = matrix(0.0, (m,n))
    v = matrix(0.0, (m,1))


    def Fkkt(W):

        # Factor 
        #
        #     S = A*D^-1*A' + I 
        #
        # where D = 2*D1*D2*(D1+D2)^-1, D1 = d[:n]**2, D2 = d[n:]**2.

        d1, d2 = W['di'][:n]**2, W['di'][n:]**2    

        # ds is square root of diagonal of D
        ds = sqrt(2.0) * div( mul( W['di'][:n], W['di'][n:]), sqrt(d1+d2) )
        d3 =  div(d2 - d1, d1 + d2)
    
        # Asc = A*diag(d)^-1/2
        blas.copy(A, Asc)
        for k in range(m):
            blas.tbsv(ds, Asc, n=n, k=0, ldA=1, incx=m, offsetx=k)

        # S = I + A * D^-1 * A'
        blas.syrk(Asc, S)
        S[::m+1] += 1.0 
        lapack.potrf(S)

        def g(x, y, z):

            x[:n] = 0.5 * ( x[:n] - mul(d3, x[n:]) + \
                    mul(d1, z[:n] + mul(d3, z[:n])) - \
                    mul(d2, z[n:] - mul(d3, z[n:])) )
            x[:n] = div( x[:n], ds) 

            # Solve
            #
            #     S * v = 0.5 * A * D^-1 * ( bx[:n] 
            #             - (D2-D1)*(D1+D2)^-1 * bx[n:] 
            #             + D1 * ( I + (D2-D1)*(D1+D2)^-1 ) * bz[:n]
            #             - D2 * ( I - (D2-D1)*(D1+D2)^-1 ) * bz[n:] )
            
            blas.gemv(Asc, x, v)
            lapack.potrs(S, v)
        
            # x[:n] = D^-1 * ( rhs - A'*v ).
            blas.gemv(Asc, v, x, alpha=-1.0, beta=1.0, trans='T')
            x[:n] = div(x[:n], ds)

            # x[n:] = (D1+D2)^-1 * ( bx[n:] - D1*bz[:n]  - D2*bz[n:] )
            #         - (D2-D1)*(D1+D2)^-1 * x[:n]         
            x[n:] = div( x[n:] - mul(d1, z[:n]) - mul(d2, z[n:]), d1+d2 )\
                    - mul( d3, x[:n] )
            
            # z[:n] = D1^1/2 * (  x[:n] - x[n:] - bz[:n] )
            # z[n:] = D2^1/2 * ( -x[:n] - x[n:] - bz[n:] ).
            z[:n] = mul( W['di'][:n],  x[:n] - x[n:] - z[:n] ) 
            z[n:] = mul( W['di'][n:], -x[:n] - x[n:] - z[n:] ) 

        return g


    x = solvers.coneqp(P, q, G, h, kktsolver = Fkkt)['x'][:n]

    x = array(x)
    return x