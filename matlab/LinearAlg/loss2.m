function loss = loss2(u,eval,W,eval_true)
if length(eval) ~= length(eval_true)
    error('Inconsistent number of eigenvectors');
end

N = size(W,1);
Dw = sum(W,2);

x = u.*sqrt(eval(:))';
xDwx = x'*(Dw.*x);
loss = trace(-2*x'*(W-Dw*Dw'/(sqrt(Dw')*sqrt(Dw)))*x/N^2 + xDwx*xDwx/N^4) + sum(eval_true.^2);
end