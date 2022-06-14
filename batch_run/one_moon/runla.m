disc = load(['./data_01/disc.mat']);

W = disc.W;
D = diag(sum(W,2));

maxepoch = 300;
d = 2; % number of nontrivial eigenvector
batchsize = 50;
% method = {'spec1_full','spec1_local','spec1_neighbor','spec2_full',...
%     'spec2_local', 'spec2_neighbor'};
% method = {'spec1_full','spec2_full'};
method = {'spec1_local'};

% eval_true1 = disc.evals(1:d+1);
eval_true2 = disc.evals(2:d+1);

lossfuncs = cell(1,1);
% lossfuncs{1} = @(u,eval)loss1(u,eval,W,eval_true1);
lossfuncs{1} = @(u,eval)loss2(u,eval,W,eval_true2);

stepsize = 0.2;
comp = disc.evecs;
for i = 1:length(method)
    la_method = method{i};
    switch la_method
        case 'spec1_full'
            [u, evals, loss, error, ss_all] = spec1_full(W, d+1, batchsize, maxepoch,...
                stepsize, lossfuncs, comp);
        case 'spec1_local'
            [u, evals, loss, error, ss_all] = spec1_local(W, d+1, batchsize, maxepoch,...
                stepsize, lossfuncs, comp);
        case 'spec1_neighbor'
            [u, evals, loss, error, ss_all] = spec1_neighbor(W, d+1, batchsize, maxepoch,...
                stepsize, lossfuncs, comp);
        case 'spec2_full'
            [u, evals, loss, error, ss_all] = spec2_full(W, d, batchsize, maxepoch,...
                stepsize, lossfuncs, comp);
        case 'spec2_local'
            [u, evals, loss, error, ss_all] = spec2_local(W, d, batchsize, maxepoch,...
                stepsize, lossfuncs, comp);
        case 'spec2_neighbor'
            [u, evals, loss, error, ss_all] = spec2_neighbor(W, d, batchsize, maxepoch,...
                stepsize, lossfuncs, comp);
    end


[evals,idx] = sort(evals,'descend');
u = u(:,idx);
save(['./data_01/LA/' la_method '-la.mat'], 'u', 'evals', 'loss', "error", 'ss_all')
end
