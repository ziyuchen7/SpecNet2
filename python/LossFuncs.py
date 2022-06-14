import numpy as np
import scipy.linalg as la

def specnet1_loss(U, evals, matdata, nettype):

    n = matdata.Size()

    W = np.double(matdata.Wmat().numpy())
    D = np.double(matdata.Dmat().numpy())
    
    if nettype == 'iterator_convert_specnet1':
        U[:,[0]] = np.multiply(1./np.sqrt(D),np.ones((n,1)))*np.sqrt(n)
        WU = np.matmul(W, U)
        DU = np.multiply(D,U)
        
        trUWU = np.sum(np.multiply(WU, U))
        trUDU = np.sum(np.multiply(DU, U))
        
        true_evals = matdata.True_Evals()
        # print(true_evals)
        loss = (trUDU - trUWU)/n**2 - U.shape[1] + np.sum(true_evals)
    
    else:
        V = np.zeros((n,U.shape[1]+1))
        V[:,1:] = U
        V[:,[0]] = np.multiply(1./np.sqrt(D),np.ones((n,1)))*np.sqrt(n)
        WV = np.matmul(W, V)
        DV = np.multiply(D,V)
        
        trVWV = np.sum(np.multiply(WV, V))
        trVDV = np.sum(np.multiply(DV, V))
        
        true_evals = matdata.True_Evals()
        # print(true_evals)
        loss = (trVDV - trVWV)/n**2 - V.shape[1] + np.sum(true_evals)
        
    return loss

def specnet2_loss(U, evals, matdata, nettype): 

    n = matdata.Size()
    
    if nettype == 'iterator_convert_specnet1':
        Y = np.multiply(U[:,1:], np.sqrt(evals[1:]))
    else:
        Y = np.multiply(U, np.sqrt(evals))

    W = np.double(matdata.Wmat().numpy())
    D = np.double(matdata.Dmat().numpy())
    Dt = np.double(matdata.Dtmat().numpy())

    WY = np.matmul(W, Y)
    DY = np.multiply(D, Y)
    DDtY = np.multiply(D, np.matmul(Dt, Y))

    trYWY = np.sum(np.multiply(WY, Y))
    YDY = np.matmul(Y.T, DY)
    trYDYYDY = np.sum(YDY*YDY)
    trYDDtY = np.sum(np.multiply(DDtY, Y))

    true_evals = matdata.True_Evals()

    loss = -2*trYWY/n**2 + 2*trYDDtY/n**2 + trYDYYDY/n**4 + np.sum(true_evals[1:]**2)
    
    return loss