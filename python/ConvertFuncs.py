import numpy as np
import scipy.linalg as la

def iterator_convert_specnet1(Y, matdata): 

    W = np.double(matdata.Wmat().numpy())
    D = np.double(matdata.Dmat().numpy())
    Y = np.double(Y.numpy())
    WY = np.matmul(W, Y)
    DY = np.multiply(D, Y)

    YWY = np.matmul(Y.T, WY)
    YDY = np.matmul(Y.T, DY)
    # YWY = (YWY + YWY.T)/2
    # YDY = (YDY + YDY.T)/2
    evals, evecs = la.eigh(YWY, YDY, driver = 'gv')
    sorted_indices = np.argsort(evals)[::-1]
    evals = evals[sorted_indices]
    evecs = evecs[:,sorted_indices]
    evecs = evecs * matdata.Size()
    U = np.matmul(Y, evecs)

    return U, evals, evecs
# def iterator_convert_specnet1(X, matdata): 

#     D = np.double(matdata.Dmat().numpy())
#     Y = np.multiply(1./np.sqrt(D),X)
#     U, evals, evecs = iterator_convert_specnet2(Y, matdata)

#     return U, evals, evecs
    
def iterator_convert_specnet2(Y, matdata): 

    W = np.double(matdata.Wmat().numpy())
    D = np.double(matdata.Dmat().numpy())
    Y = np.double(Y.numpy())
    WY = np.matmul(W, Y)
    DY = np.multiply(D, Y)

    YWY = np.matmul(Y.T, WY)
    YDY = np.matmul(Y.T, DY)
    # YWY = (YWY + YWY.T)/2
    # YDY = (YDY + YDY.T)/2
    evals, evecs = la.eigh(YWY, YDY, driver = 'gv')
    sorted_indices = np.argsort(evals)[::-1]
    evals = evals[sorted_indices]
    evecs = evecs[:,sorted_indices]
    evecs = evecs * matdata.Size()
    U = np.matmul(Y, evecs)

    return U, evals, evecs
