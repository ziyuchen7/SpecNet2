import numpy as np
import scipy.linalg as la
import torch

class MatData():
    def __init__(self, Xdata, kernel_func=None, mat=None,
            sparse_mat=None, sparse_flag=None, num_evals=0, evals = None):
        # KERNEL_FUNC is a function of two variables and return affinity
        #   matrix entries, truncation should be embedded in the function
        # MAT is a dense matrix, numpy ndarray
        # SPARSE_MAT is a sparse matrix, scipy.sparse.csr_matrix

        self.X   = torch.tensor(Xdata, dtype = torch.get_default_dtype())
        self.n   = Xdata.shape[0]
        self.dim = Xdata.shape[1]

        if kernel_func is not None:
            if sparse_flag is None:
                sparse_flag = False
            self.kernel_func = kernel_func
            if not sparse_flag:
                self.W = torch.tensor(kernel_func(Xdata, Xdata), dtype = torch.get_default_dtype())
            else:
                self.sp_nbr_idx  = []
                self.sp_val      = []
                for it in range(self.n):
                    tmp = torch.squeeze(torch.tensor(
                        kernel_func(Xdata[it,:], Xdata), dtype = torch.get_default_dtype()))
                    self.sp_nbr_idx.append(torch.LongTensor(
                        torch.nonzero(tmp,as_tuple=False)) )
                    self.sp_val.append(tmp[self.sp_nbr_idx[it]])
        elif mat is not None:
            if sparse_flag is None:
                sparse_flag = False
            if not sparse_flag:
                self.W = torch.tensor(mat, dtype = torch.get_default_dtype())
            else:
                self.sp_nbr_idx  = []
                self.sp_val      = []
                for it in range(self.n):
                    tmp = torch.squeeze(torch.tensor(mat[it,:], dtype = torch.get_default_dtype()))
                    self.sp_nbr_idx.append(torch.LongTensor(
                        torch.nonzero(tmp,as_tuple=False)) )
                    self.sp_val.append(tmp[self.sp_nbr_idx[it]])
        elif sparse_mat is not None:
            if sparse_flag is None:
                sparse_flag = True
            if not sparse_flag:
                self.W = torch.tensor(sparse_mat.toarray(), dtype = torch.get_default_dtype())
            else:
                self.sp_nbr_idx  = []
                self.sp_val      = []
                for it in range(self.n):
                    tmp = torch.squeeze(torch.tensor(
                            sparse_mat.getrow(it).toarray(), dtype = torch.get_default_dtype()))
                    self.sp_nbr_idx.append(torch.LongTensor(
                        torch.nonzero(tmp,as_tuple=False)) )
                    self.sp_val.append(tmp[self.sp_nbr_idx[it]])
        else:
            raise TypeError("Affinity matrix information is not provided.")

        self.sparse_flag = sparse_flag

        if not sparse_flag:
            self.D = torch.sum(self.W, 1)
        else:
            tmp = []
            for it in range(self.n):
                tmp.append(sum(self.sp_val[it]))
            self.D = torch.tensor(tmp, dtype = torch.get_default_dtype())
        self.D = torch.unsqueeze(self.D, -1)

        # Compute True Eigenvalues
        if not self.sparse_flag:
            if evals is None:
                self.true_evals = la.eigvalsh(self.W, np.diag(self.D.squeeze()),
                            subset_by_index=(self.n-num_evals, self.n-1))
            else:
                self.true_evals = evals[:num_evals]
        else:
            if evals is None:
                self.true_evals = la.eigvalsh(self.Wmat(),
                            np.diag(self.D.squeeze()),
                            subset_by_index=(self.n-num_evals, self.n-1))
            else:
                self.true_evals = evals[:num_evals]
        
        self.Dt = self.D/torch.matmul(torch.sqrt(self.D.t()),torch.sqrt(self.D))


    def Xdata(self, idx = None):
        if idx is None:
            return self.X
        else:
            return self.X[idx,:]

    def Wmat(self, rowidx = None, colidx = None):
        if not self.sparse_flag:
            if rowidx is None and colidx is None:
                return self.W
            if rowidx is None:
                return self.W[:,colidx]
            if colidx is None:
                return self.W[rowidx,:]
            return self.W[rowidx,:][:,colidx]
        else:
            if rowidx is None and colidx is None:
                Wtmp = torch.zeros([self.n, self.n], dtype = torch.get_default_dtype())
                for i in range(self.n):
                    Wtmp[i,self.sp_nbr_idx[i]] = self.sp_val[i]
                return Wtmp
            if rowidx is None:
                raise TypeError("Subcolumn is not supported.")
            if colidx is None:
                Wtmp = torch.zeros([len(rowidx), self.n], dtype = torch.get_default_dtype())
                for i,idx in enumerate(rowidx):
                    Wtmp[i,self.sp_nbr_idx[idx]] = self.sp_val[idx]
                return Wtmp
            Wtmp = torch.zeros([len(rowidx), self.n], dtype = torch.get_default_dtype())
            for i,idx in enumerate(rowidx):
                Wtmp[i,self.sp_nbr_idx[idx]] = self.sp_val[idx]
            return Wtmp[:,colidx]
        
    def spWmat(self, rowidx = None):
        if rowidx is None:
            return self.sp_nbr_idx, self.sp_val
        else:
            return [self.sp_nbr_idx[i] for i in rowidx], \
                [self.sp_val[i] for i in rowidx]

    def Dmat(self, idx = None):
        if idx is None:
            return self.D
        else:
            return self.D[idx,:]
        
    def Dtmat(self, idx = None):
        if idx is None:
            return self.Dt.t()
        else:
            return self.Dt[idx,:].t()

    def Wpowermat(self, rowidx = None):
        if rowidx is None:
            return self.Wpower
        else:
            return self.Wpower[rowidx,:]

    def Apowermat(self, rowidx = None):
        if rowidx is None:
            return self.Apower
        else:
            return self.Apower[rowidx,:]


    def Size(self):
        return self.n

    def Dim(self):
        return self.dim

    def True_Evals(self):
        return self.true_evals

class IndexData(torch.utils.data.Dataset):
    def __init__(self, n):
        self.n = n
    def __len__(self):
        return self.n
    def __getitem__(self, index):
        return np.array(index)