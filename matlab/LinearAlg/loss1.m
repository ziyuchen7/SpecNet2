function loss = loss1(u,eval,W,eval_true)
if length(eval) ~= length(eval_true)
    error('Inconsistent number of eigenvectors');
end

N = size(W,1);
Dw = sum(W,2);
A = (1./sqrt(Dw)).*W.*(1./sqrt(Dw))';

x = sqrt(Dw).*u;

loss = trace(x'*x - x'*A*x)/N^2 - sum(1-eval_true);
end