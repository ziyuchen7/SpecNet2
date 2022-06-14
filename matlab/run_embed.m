clear all;close all;

folderpath = '../results/mnist';
load([folderpath '/mnist_feat2.mat'])

%% pca
n = 2000;
fontsize = 18;

% x= dataX;
% mx =mean(x,1);
% x=bsxfun(@minus, x, mx);
% [u,~,~]=svd(x,'econ');
% 
% % 
% figure(1),clf;
% scatter3(u(:,1),u(:,2),u(:,3),40,labelX,'o');
% grid on; colormap(jet); colorbar();
% title('pca')

%% vis by spec
Nx = size(dataX,1);
%kx = 7;
kx = 16;
k_nn = kx*20;
[nnInds, nnDist] = knnsearch( dataX, dataX, 'k', k_nn);

%
sigma = median( nnDist(:,kx));

% build kernel matirx
rowInds = kron((1:Nx)', ones(k_nn,1));
colInds = reshape(nnInds', k_nn*Nx, 1);
vals    = reshape( exp(- nnDist.^2/(2*sigma^2))', k_nn*Nx, 1);
vals = double(vals);
K = sparse(rowInds, colInds, vals, Nx, Nx);
K = (K+K')/2;

dK = sum(K,2);

% normalize by D, "alpha=1" in diffusion map
K5 = K./((dK)*(dK)');
dK5= sum(K5,2);

%%
%disp('eig: diffusion map')
maxk = 20;

% [v,d] = eigs(-K+diag(dK), diag(dK), maxk, 'sr', 'SubspaceDimension', 50,...
%     'MaxIterations', 300, 'Display', 1);

[v,d] = eigs(-K+diag(dK), diag(dK)/n, maxk, 'sr', 'SubspaceDimension', 50,...
    'MaxIterations', 300, 'Display', 1);
d = d/Nx;

[Lambda, tmp]=sort(diag(d),'ascend');
Psi = v(:,tmp);
lambda = 1-Lambda;


%% run la
d = 4;
batchsize = 64;
maxepoch = 500;
power_order = 0;

eval_true = lambda(1:d);

lossfuncs = cell(2,1);
lossfuncs{1} = @(u,eval)loss1(u,eval,K,eval_true);
lossfuncs{2} = @(u,eval)loss2(u,eval,K,eval_true);

method = 'spec2_full';

stepsize = 2;

switch method
    case 'spec1_full'
        [u, evals, loss, ss_all] = spec1_full(K, d, batchsize, maxepoch,...
            stepsize, lossfuncs, power_order);
    case 'spec2_full'
        [u, evals, loss, ss_all] = spec2_full(K, d, batchsize, maxepoch,...
            stepsize, lossfuncs, power_order);
end
ss = median(ss_all);
[evals,idx] = sort(evals,'descend');
u = u(:,idx);

figure
plot(log10(abs(loss)));
%save([ '../results/mnist/batch_' num2str(batchsize) '/' method '_la.mat'], 'u', 'evals', 'loss', 'ss')


%% save data
x = dataX;
W = full(K);
spW = K;
evals = lambda;
sig_kernel = sigma;
evecs = Psi*sqrt(n);
%save([ '../results/mnist/disc.mat'], 'n', 'x', 'evecs', 'W', 'spW', 'evals', 'sig_kernel')


%% evaluate network output
net = load(['../results/mnist/batch_64ls/specnet2_full_unit_128_depth_2_lr_0.01.mat']);
evnet = net.y;
net_error = zeros(1,4);
for kk = 1:d
    net_error(1,kk) = norm(evnet(:,kk)-sign(evnet(:,kk)'*evecs(:,kk))*evecs(:,kk))/norm(evecs(:,kk));
end
net_error
plot(log10(abs(net.loss')));



%%
figure(2),clf;
scatter3(-Psi(:,2),Psi(:,3),-Psi(:,4),40,labelX,'o');
grid on; colormap(jet); colorbar();
set(gca, 'fontsize', fontsize);
view(30,30)

if (1)
    x = dataX;
    A = full(K);
    B = full(dK)/n;
    y = Psi(:,1:5);
    test = dataX2;
    save([folderpath '/disc.mat'], 'x', 'A', 'B', 'n', 'y')
end

nn   = load([folderpath '/y_net.mat']);
evnn   = nn.y_net;
evtest = nn.predict;
evals = diag(n-Lambda(1:5));
evdisc = Psi(:,1:5);
evdisc_sym = sqrt(B).*evdisc*sqrt(n);
true_loss1 = trace(evdisc_sym'*evdisc_sym - evdisc'*A*evdisc);
true_loss2 = trace(-2*(evdisc*sqrt(evals))'*A*(evdisc*sqrt(evals)) + ...
    ((evdisc*sqrt(evals))'*diag(B)*(evdisc*sqrt(evals)))*((evdisc*sqrt(evals))'*...
    diag(B)*(evdisc*sqrt(evals))));


if(1)
    fontsize = 35;
fpath = '/Users/ziyuchen/Desktop/SpecNet2/figure';
figure(1)
scatter3(-Psi(:,2),Psi(:,3),-Psi(:,4),40,labelX,'o');
grid on; colormap(jet); colorbar();
ylim([-0.4 0.2]);xlim([-0.2 0.3]);zlim([-0.2 0.2]);
view(30,30)
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/mnist_train_la'],'epsc')

figure(2)
scatter3(evnn(:,2),evnn(:,3),evnn(:,4),40,labelX,'o');
grid on; colormap(jet); colorbar();
ylim([-0.4 0.2]);xlim([-0.2 0.3]);zlim([-0.2 0.2]);
view(30,30)
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/mnist_train_nn'],'epsc')

figure(3)
scatter3(evtest(:,2),evtest(:,3),evtest(:,4),40,labelX2,'o');
grid on; colormap(jet); colorbar();
ylim([-0.4 0.2]);xlim([-0.2 0.3]);zlim([-0.2 0.2]);
view(30,30)
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/mnist_test_nn'],'epsc')

figure(4)
plot(1:300,log10((nn.loss1-true_loss1)/n))
hold on
plot(1:300,log10((nn.loss2-true_loss2)/n^2))
hold off
xlabel('epoch')
ylabel('$\log_{10}$','interpreter','latex')
set(gca, 'fontsize', fontsize);
legend('$\ell_1$','$\ell_2$','interpreter','latex')
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/mnist_loss'],'epsc')
end