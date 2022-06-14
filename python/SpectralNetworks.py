from torch import nn
from GradFuncs import update_YDY
import torch
from scipy.linalg import ldl
import numpy as np
class SpectralNet(nn.Module):
    def __init__(self, in_features = 5, out_features = 3,
            units = 50, depth = 4, activation = 'softplus',
            net_type = None):
            # net_type = {'specnet1', 'specnet1_ydy', 'specnet2'}
        super(SpectralNet, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.depth        = depth
        self.activation   = activation
        self.net_type     = net_type
        self.register_buffer('ortho_para', torch.eye(out_features))
        # self.register_parameter('ortho_para', nn.parameter.Parameter(torch.eye(out_features)))
        if type(units) is not list:
            self.units = [units] * (depth - 1)
        elif len(units) == 1:
            self.units = units * (depth - 1)
        else:
            self.units = units

        self.fcs = nn.ModuleList([])
        self.fcs.append(nn.Linear(in_features, self.units[0]))
        for it in range(depth-2):
            self.fcs.append(nn.Linear(self.units[it], self.units[it+1]))
        self.fcs.append(nn.Linear(self.units[depth-2], out_features))
        

        if activation.lower() == 'softplus':
            self.actfunc = nn.Softplus()
        elif activation.lower() == 'relu':
            self.actfunc = nn.ReLU()

    def forward(self, x):
        for it in range(self.depth-1):
            x = self.fcs[it](x)
            x = self.actfunc(x)
        x = self.fcs[-1](x)
        return x

    def rotate(self, x, D = None, rotate_type = '', ydy = None):
        for it in range(self.depth-1):
            x = self.fcs[it](x)
            x = self.actfunc(x)
        x = self.fcs[-1](x)
        if 'specnet1' in self.net_type.lower():
            if rotate_type.lower() == 'ydy' and ydy is not None:
                # ydy = update_YDY(GetYDY=True).detach()
                self.update_ortho_para(x = x, D = D, ydy = ydy)
            elif D is not None:
                self.update_ortho_para(x = x, D = D)
            elif rotate_type != 'same':
                self.update_ortho_para(x = x)
            # self.ortho_para = self.ortho_para.detach()
            return torch.matmul(x, self.ortho_para)
        else:
            return x

    def update_ortho_para(self, x = None, D = None, ydy = None):
        m = float(x.size()[0])
        if x is not None and D is not None and ydy is None:
            y = torch.mul(torch.sqrt(D),x)
            _, r = torch.linalg.qr(y)
            self.ortho_para = torch.inverse(r) * m
            # self.ortho_para = torch.inverse(L.t()) * m
        elif D is None and ydy is None:
            _, r = torch.linalg.qr(x, mode='reduced')
            self.ortho_para = torch.inverse(r) * np.sqrt(m)
        elif D is None and ydy is not None:
            L = torch.linalg.cholesky(ydy)
            self.ortho_para = torch.inverse(L.t()) * np.sqrt(m)
        elif ydy is not None:
            L = torch.linalg.cholesky(ydy)
            self.ortho_para = torch.inverse(L.t()) * m
        else:
            raise TypeError("At least one of y and ydy is not None")

    # def update_ortho_para(self, x = None, ydy = None):
    #     m = float(x.size()[0])
    #     if x is not None and ydy is None:
    #         _, r = torch.qr(x)
    #         self.ortho_para = torch.inverse(r) * np.sqrt(m)
    #         # self.ortho_para = nn.parameter.Parameter(torch.inverse(r) * np.sqrt(m))
    #     elif ydy is not None:
    #         L = torch.cholesky(ydy)
    #         self.ortho_para = torch.matmul(self.ortho_para,
    #                             torch.transpose(torch.inverse(L),0,1)) * np.sqrt(m)
    #     else:
    #         raise TypeError("At least one of y and ydy is not None")

