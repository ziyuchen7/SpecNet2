function spectralla(opts)

disc = load([opts.data_path '/disc.mat']);

W = disc.W;

d = opts.num_columns;
batchsize = 4;

if opts.power_order > 1
    maxepoch = 200;
else
    maxepoch = 500;
end

eval_true = disc.evals(1:d);

lossfuncs = cell(2,1);
lossfuncs{1} = @(u,eval)loss1(u,eval,W,eval_true);
lossfuncs{2} = @(u,eval)loss2(u,eval,W,eval_true);

stepsize = -1;

switch opts.method
    case 'spec1_full'
        [u, evals, loss, ss_all] = spec1_full(W, d, batchsize, maxepoch,...
            stepsize, lossfuncs, opts.power_order);
    case 'spec2_full'
        [u, evals, loss, ss_all] = spec2_full(W, d, batchsize, maxepoch,...
            stepsize, lossfuncs, opts.power_order);
end
ss = median(ss_all);

% figure(11);plot(log10(abs(losses)));
[evals,idx] = sort(evals,'descend');
u = u(:,idx);
% figure(12)
% scatter(evecs_sorted(:,1),evecs_sorted(:,2),40);
% figure(13)
% plot(ss_all)

save([opts.save_path '/la.mat'], 'u', 'evals', 'loss', 'ss')

end
