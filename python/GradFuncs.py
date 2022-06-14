import torch
import numpy as np
from functools import reduce
import time

def specnet1_grad_fullu(model, batch_idx, matdata): 
    n = matdata.Size()

    W = matdata.Wmat(rowidx=batch_idx)
    D = matdata.Dmat()
    Y = model.rotate(matdata.Xdata(), D = D)

    # DY = torch.mul(D[batch_idx], Y[batch_idx,:])
    WY = torch.matmul(W, Y)
    DinvWY = torch.mul(1./D[batch_idx], WY)
    
    grad_LA = 2*(Y[batch_idx,:] - DinvWY)
    grad_LA = grad_LA.detach()
    
    apploss = torch.trace(torch.matmul(Y[batch_idx,:].t(), grad_LA))
    apploss.backward()

def specnet1_grad_localu(model, batch_idx, matdata): 
    # start = time.time()
    n_batch = len(batch_idx)
    W_batch = matdata.Wmat(rowidx=batch_idx, colidx=batch_idx)
    D_batch = torch.sum(W_batch, 1, keepdim=True)
    Y_batch = model.rotate(matdata.Xdata(batch_idx), D=D_batch)

    # DY_batch = torch.mul(D_batch, Y_batch)
    WY_batch = torch.matmul(W_batch, Y_batch)
    DinvWY_batch = torch.mul(1./D_batch, WY_batch)

    grad_LA = 2*(Y_batch - DinvWY_batch)
    grad_LA = grad_LA.detach()

    apploss = torch.trace(torch.matmul(Y_batch.t(), grad_LA))
    apploss.backward()
    # end = time.time()
    # print(end-start)

def specnet1_grad_fake_neighboru(model, batch_idx, matdata):

    n = matdata.Size()
    n_batch = len(batch_idx)

    sp_nbr_idx_batch, sp_val_batch = matdata.spWmat(rowidx=batch_idx)
    nbr_idx = torch.LongTensor(reduce(np.union1d, sp_nbr_idx_batch))
    # nbr_idx = torch.LongTensor(range(n))

    D = matdata.Dmat()
    Z = model(matdata.Xdata())
    YDY, DY_nbr, _ = update_YDY(model, nbr_idx, Z[nbr_idx,:], matdata)
    # YDY = YDY.detach()
    Y = model.rotate(matdata.Xdata(), D = D, rotate_type = 'ydy', ydy = YDY)
    # DY_batch = torch.mul(D[batch_idx], Y[batch_idx,:])

    WY_batch = torch.zeros(n_batch, Y.shape[1])
    for i in range(n_batch): 
        WY_batch[i,:] = torch.sum(torch.mul(sp_val_batch[i],
                            Y[sp_nbr_idx_batch[i].reshape(-1),:]),
                             dim=0)
    DinvWY_batch = torch.mul(1./D[batch_idx], WY_batch)

    grad_LA = 2*(Y[batch_idx,:] - DinvWY_batch)
    grad_LA = grad_LA.detach()

    apploss = torch.trace(torch.matmul(Y[batch_idx,:].t(), grad_LA))
    apploss.backward()


## SpecNet1 loss_full uses all the data
def specnet1_grad_full(model, batch_idx, matdata): 

    W = matdata.Wmat(rowidx=batch_idx)
    D = matdata.Dmat()
    Y = model.rotate(matdata.Xdata())

    DinvY = torch.mul(1./torch.sqrt(D), Y)
    WDinvY = torch.matmul(W, DinvY)
    DinvWDinvY = torch.mul(1./torch.sqrt(D[batch_idx]), WDinvY)
    
    grad_LA = 2*(Y[batch_idx,:] - DinvWDinvY)
    grad_LA = grad_LA.detach()
    
    apploss = torch.trace(torch.matmul(Y[batch_idx,:].t(), grad_LA))
    apploss.backward()

## SpecNet1 loss_local only uses mini-batch
def specnet1_grad_local(model, batch_idx, matdata): 

    n_batch = len(batch_idx)
    Y_batch = model.rotate(matdata.Xdata(batch_idx))
    W_batch = matdata.Wmat(rowidx=batch_idx, colidx=batch_idx)
    D_batch = torch.sum(W_batch, 1, keepdim=True)

    DinvY_batch = torch.mul(1./torch.sqrt(D_batch), Y_batch)
    WDinvY_batch = torch.matmul(W_batch, DinvY_batch)
    DinvWDinvY_batch = torch.mul(1./torch.sqrt(D_batch), WDinvY_batch)

    grad_LA = 2*(Y_batch - DinvWDinvY_batch)
    grad_LA = grad_LA.detach()

    apploss = torch.trace(torch.matmul(Y_batch.t(), grad_LA))
    apploss.backward()


## SpecNet1 loss_local only uses mini-batch neighbor
def specnet1_grad_neighbor(model, batch_idx, matdata):
    
    n = matdata.Size()
    n_batch = len(batch_idx)

    sp_nbr_idx_batch, sp_val_batch = matdata.spWmat(rowidx=batch_idx)
    nbr_idx = torch.LongTensor(reduce(np.union1d, sp_nbr_idx_batch))
    batch_idx_nbr = torch.LongTensor([torch.nonzero(nbr_idx==i, \
                                    as_tuple=False) for i in batch_idx])

    D_nbr     = matdata.Dmat(nbr_idx)
    with torch.no_grad():
        Y_nbr = model(matdata.Xdata(nbr_idx))
    
    update_YDY(model, nbr_idx, Y_nbr, matdata, DisId=True)
    Y_nbr = model.rotate(matdata.Xdata(nbr_idx), rotate_type = 'ydy')

    DinvY_nbr = torch.mul(1./torch.sqrt(D_nbr), Y_nbr)

    WDinvY_batch = torch.zeros(n_batch, DinvY_nbr.shape[1])
    for i in range(n_batch): 
        idx = torch.LongTensor([torch.nonzero(nbr_idx==j, \
                            as_tuple=False) for j in sp_nbr_idx_batch[i]])
        WDinvY_batch[i,:] = torch.sum(torch.mul(sp_val_batch[i],
                            DinvY_nbr[idx,:]), dim=0)

    grad_LA = Y_nbr[batch_idx_nbr,:] - \
        torch.mul(1./torch.sqrt(D_nbr[batch_idx_nbr]), WDinvY_batch)
    grad_LA = grad_LA.detach()

    apploss = torch.trace(torch.matmul(Y_nbr[batch_idx_nbr,:].t(), grad_LA))
    apploss.backward()

def specnet1_grad_fake_neighbor(model, batch_idx, matdata):
    
    n_batch = len(batch_idx)

    sp_nbr_idx_batch, sp_val_batch = matdata.spWmat(rowidx=batch_idx)
    nbr_idx = torch.LongTensor(reduce(np.union1d, sp_nbr_idx_batch))

    D = matdata.Dmat()
    Y = model(matdata.Xdata())
    
    update_YDY(model, nbr_idx, Y[nbr_idx,:], matdata, DisId=True)
    Y = model.rotate(matdata.Xdata(), rotate_type = 'ydy')

    DinvY = torch.mul(1./torch.sqrt(D), Y)

    WDinvY_batch = torch.zeros(n_batch, DinvY.shape[1])
    for i in range(n_batch): 
        WDinvY_batch[i,:] = torch.sum(torch.mul(sp_val_batch[i],
                            DinvY[sp_nbr_idx_batch[i].reshape(-1),:]),
                             dim=0)

    grad_LA = 2*(Y[batch_idx,:] - \
        torch.mul(1./torch.sqrt(D[batch_idx]), WDinvY_batch))
    grad_LA = grad_LA.detach()

    apploss = torch.trace(torch.matmul(Y[batch_idx,:].t(), grad_LA))
    apploss.backward()

## SpecNet2 loss_full uses all the data
# def specnet2_grad_full(model, batch_idx, matdata): 

#     n = matdata.Size()
#     Y = model(matdata.Xdata())
#     W = matdata.Wmat(rowidx=batch_idx)

#     WY = torch.matmul(W, Y)
#     YDY, DY = update_YDY(model, range(matdata.Size()), Y, matdata)
    
#     grad_LA = 4*(- WY/n + torch.matmul(DY[batch_idx,:], YDY)/n**3)
#     grad_LA = grad_LA.detach()
#     apploss = torch.trace(torch.matmul(Y[batch_idx,:].t(), grad_LA))
#     apploss.backward()

def specnet2_grad_full(model, batch_idx, matdata): 

    n = matdata.Size()
    Y = model(matdata.Xdata())
    W = matdata.Wmat(rowidx=batch_idx)
    D = matdata.Dmat(batch_idx)

    WY = torch.matmul(W, Y)
    YDY, DY, DtY = update_YDY(model, range(matdata.Size()), Y, matdata)
    DDtY = torch.mul(D, DtY)
    Y_batch = model(matdata.Xdata(batch_idx))

    grad_LA = 4*(- WY/n + DDtY/n+ torch.matmul(DY[batch_idx,:], YDY)/n**3)
    grad_LA = grad_LA.detach()
    apploss = torch.trace(torch.matmul(Y_batch.t(), grad_LA))
    apploss.backward()
    
def specnet2_grad_symfull(model, batch_idx, matdata): 

    n = matdata.Size()
    Y = model(matdata.Xdata())
    W = matdata.Wmat(rowidx=batch_idx)
    D = matdata.Dmat()

    DinvY = torch.mul(1./torch.sqrt(D), Y)
    WDinvY = torch.matmul(W, DinvY)
    DinvWDinvY = torch.mul(1./torch.sqrt(D[batch_idx]), WDinvY)
    
    YY, _, _ = update_YDY(model, range(matdata.Size()), Y, matdata, DisId=True)
    v1 = torch.ones(n,1)/np.sqrt(n)
    DDtY = torch.mul(v1[batch_idx,:], torch.matmul(v1.t(),Y))

    grad_LA = 4*(- DinvWDinvY + DDtY+ torch.matmul(Y[batch_idx,:], YY)/n)
    grad_LA = grad_LA.detach()
    apploss = torch.trace(torch.matmul(Y[batch_idx,:].t(), grad_LA))
    apploss.backward()
    
## SpecNet2 loss_local only uses mini-batch
def specnet2_grad_local(model, batch_idx, matdata):

    n_batch = len(batch_idx)
    Y_batch = model(matdata.Xdata(batch_idx))
    W_batch = matdata.Wmat(rowidx=batch_idx, colidx=batch_idx)
    D_batch = torch.sum(W_batch, 1, keepdim=True)
    Dt      = D_batch.t()/torch.matmul(torch.sqrt(D_batch.t()),torch.sqrt(D_batch))
    DDtY    = torch.mul(D_batch, torch.matmul(Dt, Y_batch))
    WY_batch = torch.matmul(W_batch, Y_batch)
    DY_batch = torch.mul(D_batch, Y_batch)
    YDY_batch = torch.matmul(Y_batch.t(), DY_batch)
    
    grad_LA = 4*(- WY_batch/n_batch + DDtY/n_batch\
        + torch.matmul(DY_batch, YDY_batch)/n_batch**3)
    grad_LA = grad_LA.detach()

    apploss = torch.trace(torch.matmul(Y_batch.t(), grad_LA))
    apploss.backward()
    

def specnet2_grad_neighboru(model, batch_idx, matdata):
    start = time.time()
    n = matdata.Size()
    n_batch = len(batch_idx)
    start4 = time.time()
    D = matdata.Dmat(batch_idx)
    end4 = time.time()
    start5 = time.time()
    sp_nbr_idx_batch, sp_val_batch = matdata.spWmat(rowidx=batch_idx)
    nbr_idx = torch.LongTensor(reduce(np.union1d, sp_nbr_idx_batch))
    
    batch_idx_nbr = torch.LongTensor([torch.nonzero(nbr_idx==i, \
                                    as_tuple=False) for i in batch_idx])
    end5 = time.time()
    start2 = time.time()
    Y_nbr  = model(matdata.Xdata(nbr_idx))
    Y_batch = model(matdata.Xdata(batch_idx))
    end2 = time.time()
    start1 = time.time()
    W_batch = matdata.Wmat(rowidx=batch_idx, colidx=nbr_idx)
    WY_batch = torch.matmul(W_batch, Y_nbr)
    end1 = time.time()
    start6 = time.time()
    YDY, DY_nbr, DtY = update_YDY(model, nbr_idx, Y_nbr, matdata) 
    
    DDtY = torch.mul(D, DtY)
    grad_LA = 4*(- WY_batch/n + DDtY/n \
        + torch.matmul(DY_nbr[batch_idx_nbr,:], YDY)/n**3)
    grad_LA = grad_LA.detach()
    end6 = time.time()
    apploss = torch.trace(torch.matmul(Y_batch.t(), grad_LA))
    start3 = time.time()
    apploss.backward()
    end3 = time.time()
    return end1-start1, end2-start2, end3-start3, end3-start, end4-start4, end5-start5, end6-start6
## SpecNet2 loss_local only uses mini-batch neighbor
def specnet2_grad_neighbor(model, batch_idx, matdata):
    n = matdata.Size()
    n_batch = len(batch_idx)

    D = matdata.Dmat(batch_idx)

    sp_nbr_idx_batch, sp_val_batch = matdata.spWmat(rowidx=batch_idx)
    nbr_idx = torch.LongTensor(reduce(np.union1d, sp_nbr_idx_batch))
    nbr_idx = nbr_idx.reshape(-1)
    batch_idx_nbr = torch.LongTensor([torch.nonzero(nbr_idx==i, \
                                    as_tuple=False) for i in batch_idx])

    Y_nbr  = model(matdata.Xdata(nbr_idx))
    Y_batch = model(matdata.Xdata(batch_idx))

    W_batch = matdata.Wmat(rowidx=batch_idx, colidx=nbr_idx)
    WY_batch = torch.matmul(W_batch, Y_nbr)

    YDY, DY_nbr, DtY = update_YDY(model, nbr_idx, Y_nbr, matdata)

    DDtY = torch.mul(D, DtY)

    grad_LA = 4*(- WY_batch/n + DDtY/n \
        + torch.matmul(DY_nbr[batch_idx_nbr,:], YDY)/n**3)
    grad_LA = grad_LA.detach()
    
    apploss = torch.trace(torch.matmul(Y_batch.t(), grad_LA))
    apploss.backward()

def specnet2_grad_fake_neighbor(model, batch_idx, matdata):
    n = matdata.Size()
    n_batch = len(batch_idx)
    
    D = matdata.Dmat(batch_idx)
    Dt = matdata.Dtmat()

    sp_nbr_idx_batch, sp_val_batch = matdata.spWmat(rowidx=batch_idx)
    nbr_idx = torch.LongTensor(reduce(np.union1d, sp_nbr_idx_batch))
    batch_idx_nbr = torch.LongTensor([torch.nonzero(nbr_idx==i, \
                                    as_tuple=False) for i in batch_idx])
    Y  = model(matdata.Xdata())
    Y_batch = model(matdata.Xdata(batch_idx))

    WY_batch = torch.zeros(n_batch, Y.shape[1])
    for i in range(n_batch):
        WY_batch[i,:] = torch.sum(torch.mul(sp_val_batch[i], Y[sp_nbr_idx_batch[i].reshape(-1),:]),
                            dim=0)

    YDY, DY_nbr, Y_part = update_YDY(model, nbr_idx, Y[nbr_idx,:], matdata)

    DDtY = torch.mul(D, torch.matmul(Dt, Y_part))

    grad_LA = 4*(- WY_batch/n + DDtY/n \
        + torch.matmul(DY_nbr[batch_idx_nbr,:], YDY)/n**3)
    grad_LA = grad_LA.detach()

    apploss = torch.trace(torch.matmul(Y_batch.t(), grad_LA))
    apploss.backward()
  
    
# def specnet2_grad_fake_neighbor(model, batch_idx, matdata):
    
#     n = matdata.Size()
#     n_batch = len(batch_idx)

#     sp_nbr_idx_batch, sp_val_batch = matdata.spWmat(rowidx=batch_idx)
#     nbr_idx = torch.LongTensor(reduce(np.union1d, sp_nbr_idx_batch))
#     batch_idx_nbr = torch.LongTensor([torch.nonzero(nbr_idx==i, \
#                                     as_tuple=False) for i in batch_idx])

#     Y  = model(matdata.Xdata())

#     WY_batch = torch.zeros(n_batch, Y.shape[1])
#     for i in range(n_batch):
#         WY_batch[i,:] = torch.sum(torch.mul(sp_val_batch[i], Y[sp_nbr_idx_batch[i].reshape(-1),:]),
#                             dim=0)

#     YDY, DY_nbr = update_YDY(model, nbr_idx, Y[nbr_idx,:], matdata)

#     grad_LA = 4*(- WY_batch/n \
#         + torch.matmul(DY_nbr[batch_idx_nbr,:], YDY)/n**3)
#     grad_LA = grad_LA.detach()

#     apploss = torch.trace(torch.matmul(Y[batch_idx,:].t(), grad_LA))
#     apploss.backward()

## Updating YDY approximately
def update_YDY(model = None, idx = None, Y_idx = None, matdata = None,
                DisId=False, GetYDY=False):
    
    if update_YDY.Y_old is None:
        model.eval()
        with torch.no_grad():
            update_YDY.Y_old = model(matdata.Xdata())
            update_YDY.Y_old = update_YDY.Y_old.detach()
        if not DisId:
            D = matdata.Dmat()
            Dt = matdata.Dtmat()
            update_YDY.YDY = torch.matmul(
                update_YDY.Y_old.t(), torch.mul(D,
                update_YDY.Y_old))
            update_YDY.DtY = torch.matmul(Dt, update_YDY.Y_old)
        else:
            update_YDY.YDY = torch.matmul(
                update_YDY.Y_old.t(), update_YDY.Y_old)
    
    if GetYDY:
        return update_YDY.YDY
    
    Y_idx = Y_idx.detach()
    Y_old_idx = update_YDY.Y_old[idx,:]
    # Y_idx = Y_idx.detach()

    if not DisId:
        D_idx = matdata.Dmat(idx)
        Dt_idx = matdata.Dtmat(idx)
        DY_idx     = torch.mul(D_idx, Y_idx)
        DY_old_idx = torch.mul(D_idx, Y_old_idx)
        DtY_idx     = torch.matmul(Dt_idx, Y_idx)
        DtY_old_idx = torch.matmul(Dt_idx, Y_old_idx)
    else:
        DY_idx     = Y_idx
        DY_old_idx = Y_old_idx

    update_YDY.YDY = update_YDY.YDY \
        - torch.matmul(Y_old_idx.t(), DY_old_idx) \
        + torch.matmul(Y_idx.t(), DY_idx)

    update_YDY.DtY = update_YDY.DtY \
        - DtY_old_idx + DtY_idx
        
    update_YDY.Y_old[idx,:] = Y_idx
    # update_YDY.YDY = update_YDY.YDY \
    #     - torch.matmul(Y_old_idx.t(), DY_old_idx) \
    #     + torch.matmul(Y_idx.t(), DY_idx)

    # update_YDY.Y_old[idx,:] = Y_idx

    return update_YDY.YDY, DY_idx, update_YDY.DtY

update_YDY.Y_old = None
update_YDY.YDY = None
update_YDY.DtY = None