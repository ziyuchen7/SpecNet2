function [u, eval, losses, error, ss_all] = spec2_full(W_old, d, batchsize, maxepoch, stepsize, lossfuncs, evcont)
% SpectralNet2 LA Version

N = size(W_old,1);
Dw = sum(W_old,2);
losses = zeros(maxepoch,1);
x = randn(N,d);
ss_all = zeros(maxepoch,1);
W = W_old-Dw*Dw'/(sqrt(Dw')*sqrt(Dw));
error = zeros(maxepoch,d);

for epoch = 1:maxepoch
    [batchidx, numbatch] = gen_batch(N, batchsize);
    xDwx = x'*(Dw.*x);
    for it = 1:numbatch
        idx = batchidx{it};
        %nbr_idx = find(sum(abs(W(idx,:))));
        nbr_idx = 1:N;
        %xDwx = (x(nbr_idx,:)'*(Dw(nbr_idx).*x(nbr_idx,:)))/length(nbr_idx)*N;
        g = 4* (-W(idx,nbr_idx)*x(nbr_idx,:)/N + Dw(idx).*x(idx,:)*xDwx/N^3);
        if stepsize < 0
            gDwg = g'*(Dw(idx).*g);
            xDwg = x(idx,:)'*(Dw(idx).*g);
            gWg = g'*W(idx,idx)*g;
            xWg = x(nbr_idx,:)'*W(nbr_idx,idx)*g;
            ss = spec2_ls(gWg/N^2,xWg/N^2,gDwg/N^2,xDwg/N^2,xDwx/N^2);
        else
            ss = stepsize;
        end
        xDwx = xDwx ...
            - ss*g'*(Dw(idx).*x(idx,:)) - ss*x(idx,:)'*(Dw(idx).*g) ...
            + ss^2*g'*(Dw(idx).*g);
        x(idx,:) = x(idx,:) - ss * g;
    end
    [u,eval] = spec2convert(x, W_old);
    [eval,idx] = sort(eval,'descend');
    u = u(:,idx);
    for num = 1:d
        coef = u(:,num)'*evcont(:,1+num)/(u(:,num)'*u(:,num));
        error(epoch,num) = norm(coef*u(:,num)-evcont(:,1+num))/norm(evcont(:,1+num));
    end
    ss_all(epoch) = ss;
    fprintf("Iter %4d: ", epoch);
    for it = 1:length(lossfuncs)
        losses(epoch,it) = lossfuncs{it}(u,eval);
        fprintf(" %10e ", losses(epoch,it));
    end
    fprintf("\n");
end