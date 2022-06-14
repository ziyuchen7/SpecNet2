disc = load(['./data_01/disc.mat']);

n = 2000;
x = randn(2000,3);
W = disc.W;
D = diag(sum(W,2));
d = 2;
epoch = 3000;
eval_true2 = disc.evals(2:d+1);

lossfuncs = cell(1,1);

lossfuncs{1} = @(u,eval)loss2(u,eval,W,eval_true2);
DW = D^(-0.5)*W*D^(-0.5);
% y = randn(2000,3);
y = D^(-0.5)*x;
for i = 1:epoch
    y = W*y;
    [Q,R] = qr(D^(-0.5)*y,0);
    y = D^(-0.5)*Q;
    y'*W*y;
    
    x = DW*x;
    [Q,R] = qr(x,0);
    x = Q;
    x'*DW*x;
    y'*W*y-x'*DW*x
end




% for i = 1:epoch
%     
%     x = DW*x;
%     [Q,R] = qr(x,0);
%     x = Q;
%     x'*DW*x
% %     [u,eval] = spec1convert(x, W);
% %     [eval,idx] = sort(eval,'descend');
% %     u = u(:,idx);
% %     loss2(u(:,2:end),eval(2:end),W,eval_true2)
% %     eval
% end

