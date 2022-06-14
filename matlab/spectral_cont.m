function [evecs, evals] = spectral_cont(x,p,sig_kernel,kmax)

idx_nnz = p > 1e-6*max(p);
p = p(idx_nnz);

Kernel = gaufunc(x(idx_nnz,:),x(idx_nnz,:),sig_kernel);

int_kernel = (Kernel.*p).*p';

degree = sum(int_kernel,2)/size(idx_nnz,1);

[ev,evals] = eigs( int_kernel, diag(degree), kmax);

evals = evals/size(idx_nnz,1);
ev = ev/sqrt(size(idx_nnz,1));

evecs = zeros(size(x,1),kmax);
evecs(idx_nnz,:) = ev(:,:);

end