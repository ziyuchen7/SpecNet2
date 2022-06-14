 function [u,eval] = spec2convert(x, W)
% input: x'*Dw*x = N*lambda
% output: generalized eigenvalue and eigenvector for W and Dw such that
% u'*Dw*u = N^2*I

N = size(W,1);
Dw = sum(W,2);

xWx = x'*W*x;
xDwx = x'*(Dw.*x);

[ev,eval] = eig_sym(xWx, xDwx);
eval = diag(eval);

u = N*x*ev;
end