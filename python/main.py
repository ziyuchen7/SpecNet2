import sys
import json
import scipy.io
import numpy as np
import torch
from torch import optim

import ConvertFuncs
from Datasets import *
import GradFuncs
from LossFuncs import *
from SpectralNetworks import *
from Training import *
import tracemalloc
torch.set_default_dtype(torch.float64)
json_file = open(sys.argv[1])
# json_file = open('mnist_para.json')
paras = json.load(json_file)

nnparas = paras['neural network']
dsparas = paras['data set']
ttparas = paras['train and test']
nfparas = paras['net function']
flag    = paras['classify_flag']
# torch.manual_seed(1)

# Prepare affinity matrix data
matlab_data = scipy.io.loadmat(dsparas['path'])
Xdata = matlab_data['x']
# mat   = matlab_data['W']
spmat = matlab_data['spW']
if 'evals' in dict.keys(matlab_data):
    matevals = matlab_data['evals']
else:
    matevals = None
if flag:
    label = matlab_data['label']
    evcont = None
else:
    evcont = matlab_data['evcont']
    label = None
test_data = scipy.io.loadmat(dsparas['tpath'])
Tdata = torch.tensor(test_data['x'], dtype = torch.get_default_dtype())
if nfparas['net type'].lower() == 'specnet1':
    num_evals = nnparas['output dimension']
else:
    num_evals = nnparas['output dimension'] + 1
matdata = MatData(Xdata=Xdata, sparse_mat=spmat,
                        num_evals=num_evals, evals = matevals)
net_type = nfparas['net type'].lower() 
if 'neighbor' in nfparas['gradient type'].lower():
    net_type = net_type + '_ydy'


# Prepare model structure, optimizer, and loss functions
model = SpectralNet(
            in_features = matdata.Dim(),
            out_features = nnparas['output dimension'],
            units = nnparas['units'], depth = nnparas['depth'],
            activation = nnparas['activation'], net_type = net_type)

# for name, param in model.named_parameters():
#     print (name, param.data)
# for name, buffer in model.named_buffers():
#     print (name, buffer.data)

# with torch.no_grad():
#     Y = model(matdata.Xdata()).detach()
#     print(Y)
#     print(torch.matmul(Y.t(),Y))


optimizer = optim.Adam(model.parameters(), lr=ttparas['learning rate'])
# optimizer = optim.RMSprop(model.parameters(), lr=ttparas['learning rate'])
# optimizer = optim.Adagrad(model.parameters(), lr=ttparas['learning rate'])
# optimizer = optim.SGD(model.parameters(), lr=ttparas['learning rate'])
gradfunc_type = nfparas['net type'].lower() + '_grad_' \
                + nfparas['gradient type'].lower()
gradfunc = getattr(GradFuncs, gradfunc_type)

convertfunc_type = 'iterator_convert_' + nfparas['net type'].lower()
convertfunc = getattr(ConvertFuncs, convertfunc_type)

lossfuncs = [specnet1_loss, specnet2_loss]

losses_hist1, losses_hist2, losses_hist3 = train(model, optimizer, matdata, ttparas, 
                        gradfunc, convertfunc, lossfuncs, evcont, test_data, flag, label)

# for name, param in model.named_parameters():
#     print (name, param.data)
# for name, buffer in model.named_buffers():
#     print (name, buffer.data)

with torch.no_grad():
    Y = model.rotate(matdata.Xdata(),rotate_type = 'same').detach()
    U, evals, evecs = convertfunc(Y, matdata)
    testY = model.rotate(Tdata, rotate_type = 'same').detach()
    U_test = np.matmul(testY.numpy(),evecs)


scipy.io.savemat(paras['log path']+ '/' + nfparas['net type'].lower() +
                 '_' + nfparas['gradient type'].lower() + '_unit_' + 
                 str(nnparas['units']) + '_depth_' + str(nnparas['depth']) + '_lr_' 
                 + str(ttparas['learning rate']) + '_batch_' + str(ttparas['batch size']) +
                 '-'+ str(ttparas['data order']) + '.mat',
        {'grad_loss':losses_hist1, 'test_loss':losses_hist2, 'train_loss': losses_hist3, 'y': U, 'testy': U_test, 'evals':evals})