function [u, eval, losses, error, ss_all] = spec1_neighbor(W, d, batchsize, maxepoch, stepsize, lossfuncs, evcont)
% SpectralNet LA Version

N = size(W,1);
Dw = sum(W,2);
error = zeros(maxepoch,d);

losses = zeros(maxepoch,length(lossfuncs));
x = randn(N,d);
ss_all = zeros(maxepoch,1);

for epoch = 1:maxepoch
    [batchidx, numbatch] = gen_batch(N, batchsize);
    for it = 1:numbatch
        idx = batchidx{it};
        nbr_idx = find(sum(abs(W(idx,:))));
        Wx = W(idx,nbr_idx)*x(nbr_idx,:);
        g = 2*(x(idx,:) - 1./Dw(idx).*Wx);
        if stepsize < 0
            ss = trace(g'*x(idx,:)-g'*Ax)/trace(g'*g - g'*A(idx,idx)*g);
            if abs(trace(g'*g - g'*A(idx,idx)*g)) < 1e-14
                ss = 0.1;
                fprintf('Active %3d',epoch);
            end
        else
            ss = stepsize;
        end
        x(idx,:) = x(idx,:) - ss * g;
        [~,R] = qr(sqrt(Dw(nbr_idx)).*x(nbr_idx,:),0);
        x = x/R*N;
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
