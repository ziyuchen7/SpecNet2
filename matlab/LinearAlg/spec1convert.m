function [u,eval] = spec1convert(x, W)
% input: x'*x = N*I
% output: generalized eigenvalue and eigenvector for W and dW such that
% u'*Dw*u = N^2*I

N = size(W,1);
Dw = sum(W,2);

xWx = x'*W*x;
xDwx = x'*(Dw.*x);

[ev,eval] = eig_sym(xWx, xDwx);
eval = diag(eval);

u = N*x*ev;
end