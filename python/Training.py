import torch
from numpy.linalg import norm 
from numpy import inner
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from itertools import permutations
from Datasets import *
from GradFuncs import update_YDY
import matplotlib.pyplot as plt
import scipy.io
import time

def train_epoch(model, optimizer, matdata, idxdata, paras, gradfunc):
    for it_batch, (batch_idx) in enumerate(idxdata):
        optimizer.zero_grad() 
        gradfunc(model, batch_idx, matdata)
        optimizer.step()
    
    return

def train(model, optimizer, matdata, paras,
        gradfunc, convertfunc, lossfuncs, evcont, test_data, flag, label):

    idxdata = torch.utils.data.DataLoader( IndexData(matdata.Size()),
            batch_size = paras['batch size'], shuffle=True)

    losses_hist1 = [[] for _ in range(len(lossfuncs))]
    Tdata = torch.tensor(test_data['x'], dtype = torch.get_default_dtype())
    
    if not flag:
        if convertfunc.__name__ == 'iterator_convert_specnet2':
            Tevcont = test_data['evcont']
            dim = Tevcont.shape[1]-1
            evcont = evcont[:, :dim+1]
            losses_hist2 = [[] for _ in range(dim)]
            losses_hist3 = [[] for _ in range(dim)]
        else:
            Tevcont = test_data['evcont']
            dim = Tevcont.shape[1]
            evcont = evcont[:,:dim]
            losses_hist2 = [[] for _ in range(dim)]
            losses_hist3 = [[] for _ in range(dim)]
    else:
        if convertfunc.__name__ == 'iterator_convert_specnet2':
            Tlabel = test_data['label']
            losses_hist2 = [[] for _ in range(1)]
            losses_hist3 = [[] for _ in range(1)]
        else:
            Tlabel = test_data['label']
            losses_hist2 = [[] for _ in range(1)]
            losses_hist3 = [[] for _ in range(1)]
    
    # optimizer.zero_grad()
    for it_epoch in range(paras['max num of epochs']):

        # if it_epoch > 50:
        #     optimizer.param_groups[0]['lr'] = 1e-3
        
        # if it_epoch > 300:
        #     optimizer.param_groups[0]['lr'] = 1e-3

        train_epoch(model, optimizer, matdata, idxdata, paras, gradfunc)



        with torch.no_grad():
            Y = model.rotate(matdata.Xdata(),rotate_type = 'same').detach()
            testY = model.rotate(Tdata,rotate_type = 'same').detach()
        
        if convertfunc.__name__ == 'iterator_convert_specnet2':
            U, evals, evecs = convertfunc(Y, matdata)
            testU = np.matmul(testY,evecs)
            if not flag:
                for num in range(dim):
                    losses_hist2[num].append( fun_error(testU[:,num], Tevcont[:,num+1]) )
                    losses_hist3[num].append( fun_error(U[:,num], evcont[:,num+1]) )
            elif flag == 1:
                kmeans3 = KMeans(n_clusters=2).fit(U)
                test_predict = kmeans3.predict(testU)
                losses_hist2[0].append( classify_error(test_predict, Tlabel.reshape(-1)) )
                losses_hist3[0].append( classify_error(kmeans3.labels_, label.reshape(-1)) )
            # elif flag == 2:
            #     kmeans3 = KMeans(n_clusters=5).fit(U)
            #     test_predict = kmeans3.predict(testU)
            #     losses_hist2[0].append( mnist_error(test_predict, Tlabel.reshape(-1),5) )
            #     losses_hist3[0].append( mnist_error(kmeans3.labels_, label.reshape(-1),5) )
        elif convertfunc.__name__ == 'iterator_convert_specnet1':
            U, evals, evecs = convertfunc(Y, matdata)
            testU = np.matmul(testY,evecs)
            if not flag:
                for num in range(dim):
                    losses_hist2[num].append( fun_error(testU[:,num], Tevcont[:,num]) )
                    losses_hist3[num].append( fun_error(U[:,num], evcont[:,num]) )
            elif flag == 1:
                kmeans3 = KMeans(n_clusters=2).fit(Y[:,1].reshape(-1,1))
                test_predict = kmeans3.predict(testY[:,1].reshape(-1,1))
                losses_hist2[0].append( classify_error(test_predict, Tlabel.reshape(-1)) )
                losses_hist3[0].append( classify_error(kmeans3.labels_, label.reshape(-1)) )
            # elif flag == 2:
            #     kmeans3 = KMeans(n_clusters=2).fit(Y)
            #     test_predict = kmeans3.predict(testY)
            #     losses_hist2[0].append( mnist_error(test_predict, Tlabel.reshape(-1),5) )
            #     losses_hist3[0].append( mnist_error(kmeans3.labels_, label.reshape(-1),5) )

        for it_loss, lossfunc in enumerate(lossfuncs):
            losses_hist1[it_loss].append( lossfunc(U, evals, matdata, convertfunc.__name__) )
        

        print('====> Epoch: {:6d} Trace loss: {:8e},    {:8e}'.format(
            it_epoch, losses_hist1[0][it_epoch],losses_hist1[1][it_epoch]))
        if losses_hist1[0][it_epoch] < paras['tolerance']:
            break
    return losses_hist1, losses_hist2, losses_hist3


def fun_error(evnet, evcont):
    a = inner(evnet,evcont)/inner(evnet,evnet)
    relative_error = norm(a*evnet-evcont)/norm(evcont)

    return relative_error

def classify_error(netlabel, Tlabel):
    netlabel += 1
    score = np.mean(abs(netlabel - Tlabel))
    score = max(score, 1-score)
    
    return score
