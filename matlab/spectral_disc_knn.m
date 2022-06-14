function [Kernel, degree, evecs, evals, I, J, V] = spectral_disc_knn(x, sig_kernel, kmax, knn)

n = size(x,1);

[idx,dis]=knnsearch(x,x,'k',knn);
%idx=idx(:,2:end);
%dis=dis(:,2:end);


I= kron((1:n)',ones(knn,1));
J= reshape(idx',[n*knn,1]);
V = reshape(exp(-dis.^2/(2*sig_kernel^2))'/(2*pi*sig_kernel^2), [n*knn,1]);
Kernel=sparse(I,J,V,n,n);
Kernel=max(Kernel,Kernel');

degree = sum(Kernel,2)/n;

[evecs,evals] = eigs(Kernel, diag(degree), kmax);
end