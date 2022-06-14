function [u, eval, losses, error, ss_all] = spec2_local(W, d, batchsize, maxepoch, stepsize, lossfuncs, evcont)
% SpectralNet2 LA Version

N = size(W,1);
losses = zeros(maxepoch,1);
x = randn(N,d);
ss_all = zeros(maxepoch,1);
error = zeros(maxepoch,d);

for epoch = 1:maxepoch
    [batchidx, numbatch] = gen_batch(N, batchsize);
    for it = 1:numbatch
        idx = batchidx{it};
        Wsub = W(idx,idx);
        Dwsub = sum(Wsub,2);
        Wsub = Wsub-Dwsub*Dwsub'/(sqrt(Dwsub')*sqrt(Dwsub));
        n = length(idx);
        
        xDwxsub = x(idx,:)'*(Dwsub.*x(idx,:));
        g = 4* (-Wsub*x(idx,:)/n + Dwsub.*x(idx,:)*(xDwxsub)/n^3);
        if stepsize < 0
            gDwgsub = g'*(Dwsub.*g);
            xDwgsub = x(idx,:)'*(Dwsub.*g);
            gWgsub = g'*Wsub*g;
            xWgsub = x(idx,:)'*Wsub*g;
            ss = spec2_ls(gWgsub,xWgsub,gDwgsub/n,xDwgsub/n,xDwxsub/n);
        else
            ss = stepsize;
        end
        x(idx,:) = x(idx,:) - ss * g;
    end
    [u,eval] = spec2convert(x, W);
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
