function [Kernel, degree, evecs, evals] = spectral_disc(x, sig_kernel, kmax)

n = size(x,1);

Kernel = gaufunc(x,x,sig_kernel);
% idx = Kernel>0.3;
% Kernel = Kernel.*idx;

degree = sum(Kernel,2);

[evecs,evals] = eigs(Kernel, diag(degree), kmax);

end