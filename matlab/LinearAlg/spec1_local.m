function [u, eval, losses, error, ss_all] = spec1_local(W, d, batchsize, maxepoch, stepsize, lossfuncs, evcont)
% SpectralNet LA Version

N = size(W,1);
losses = zeros(maxepoch,length(lossfuncs));
error = zeros(maxepoch,d);
x = randn(N,d);
ss_all = zeros(maxepoch,1);

for epoch = 1:maxepoch
    [batchidx, numbatch] = gen_batch(N, batchsize);
    for it = 1:numbatch
        idx = batchidx{it};
        Wsub = W(idx,idx);
        Dwsub = sum(Wsub,2);
        Wsubx = W(idx,idx)*x(idx,:);
        g = 2*(x(idx,:) - 1./Dwsub.*Wsubx);
        if stepsize < 0
            ss = trace(g'*x(idx,:)-g'*Axsub)/trace(g'*g - g'*Asub*g);
            if abs(trace(g'*g - g'*Asub*g)) < 1e-10
                ss = 0.1;
            end
        else
            ss = stepsize;
        end
        x(idx,:) = x(idx,:) - ss * g;
        [~,R] = qr(sqrt(Dwsub).*x(idx,:),0);
        x = x/R*length(idx);
    end
    [u,eval] = spec1convert(x, W);
    [eval,idx] = sort(eval,'descend');
    u = u(:,idx);
    for num = 1:d
        coef = u(:,num)'*evcont(:,num)/(u(:,num)'*u(:,num));
        error(epoch,num) = norm(coef*u(:,num)-evcont(:,num))/norm(evcont(:,num));
    end
    ss_all(epoch) = ss;
    fprintf("Iter %4d: ", epoch);
    for it = 1:length(lossfuncs)
        losses(epoch,it) = lossfuncs{it}(u(:,2:d),eval(2:d));
        fprintf(" %10e ", losses(epoch,it));
    end
    fprintf("\n");
end
