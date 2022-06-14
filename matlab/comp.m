clear all
close all
rng(2020)

folderpath = '../results/sparse_one_moon';
cont = load([folderpath '/cont.mat']);
disc = load([folderpath '/disc.mat']);

n  = disc.n;
kmax = cont.kmax;

x1  = disc.x(:,1);
x2  = disc.x(:,2);
x1grid = cont.x1grid;
x2grid = cont.x2grid;
evcont = zeros(n, kmax);

for it = 1:kmax
  evcont(:,it) = interp2(x1grid, x2grid, ...
    reshape(cont.evecs(:,it),size(x1grid)), x1, x2, 'spline');
end

if (0)
    save([folderpath '/evcont.mat'], 'evcont')
end


nn   = load([folderpath '/y_net.mat']);
spe  = load([folderpath '/spe_net.mat']);

evdisc = disc.y;

evnn   = nn.y_net;
evspe   = spe.y_net;

if (1) % activate in two moons case
    [rot1] = rot_matrix(evnn);
    evnn = evnn * rot1;
    [rot2] = rot_matrix(evspe);
    evspe = evspe * rot2;
    [rot3] = rot_matrix(evcont);
    evcont = evcont * rot3;
end

%evspe = flip(evspe,2);
% Left for comparison among evcont, evdisc, evnn
error_disc = zeros(2,1);
error_spe = zeros(2,1);
error_nn = zeros(2,1);
evcont = evcont./sqrt(diag(evcont'*diag(disc.B)*evcont))'*n;

for i = 1:2
    error_disc(i) = norm(evdisc(:,i)-sign(evdisc(:,i)'*evcont(:,i))*evcont(:,i))/norm(evcont(:,i));
    error_spe(i) = norm(evspe(:,i)-sign(evspe(:,i)'*evcont(:,i))*evcont(:,i))/norm(evcont(:,i));
    error_nn(i) = norm(evnn(:,i)-sign(evnn(:,i)'*evcont(:,i))*evcont(:,i))/norm(evcont(:,i));
end

% Check sparsity
spar = sum(sum(disc.A~=0,2))/n;


% perform spectral clustering
idx0 = kmeans(evdisc,2);
idx1 = kmeans(evnn,2);
idx2 = kmeans(evspe,2);

score0 = mean(abs(idx0 - [2*ones(n/2,1);ones(n/2,1)]));
score1 = mean(abs(idx1 - [2*ones(n/2,1);ones(n/2,1)]));
score2 = mean(abs(idx2 - [2*ones(n/2,1);ones(n/2,1)]));
score0 = max(score0, 1-score0)
score1 = max(score1, 1-score1)
score2 = max(score2, 1-score2)

%evdisc_sym = sqrt(disc.B).*evdisc*sqrt(n);
%true_loss1 = trace(evdisc_sym'*evdisc_sym - evdisc'*disc.A*evdisc);
W = disc.A;
dW = sum(W,2);
A = diag(1./sqrt(dW))*W*diag(1./sqrt(dW));
evsum = sum(eigs(A,2));
true_loss1 = (2 - evsum)*n;
true_loss2 = trace(-2*(evdisc*sqrt(disc.evals))'*disc.W*(evdisc*sqrt(disc.evals))/n^2 + ...
    ((evdisc*sqrt(disc.evals))'*diag(sum(disc.W,2))*(evdisc*sqrt(disc.evals)))*((evdisc*sqrt(disc.evals))'*...
    diag(sum(disc.W,2))*(evdisc*sqrt(disc.evals)))/n^4);


%% generate figures
fontsize = 45;

fpath = '/Users/ziyuchen/Desktop/SpecNet2/figure';
if(1)
figure(1);clf
plot(-evspe(:,2))
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/16twomoon_specnet1_evec'],'epsc')

figure(2);clf
plot(-evnn(:,2))
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/16twomoon_specnet2_evec'],'epsc')

figure(3);clf
plot(1:300,log10((spe.loss1 - true_loss1)/n),'LineWidth',2)
hold on
plot(1:300,log10((nn.loss1 - true_loss1)/n),'LineWidth',2)
ylim([-5 -1])
xlabel('epoch')
ylabel('$\log_{10}\ell_1$','interpreter','latex')
set(gca, 'fontsize', fontsize);
legend('SpecNet1','SpecNet2','location','east')
saveas(gcf,[fpath '/16twomoon_loss1'],'epsc')

figure(4);clf
plot(1:300,log10((spe.loss2 - true_loss2)/n^2),'LineWidth',2)
hold on
plot(1:300,log10((nn.loss2 - true_loss2)/n^2),'LineWidth',2)
ylim([-5 -1])
xlabel('epoch')
ylabel('$\log_{10}\ell_2$','interpreter','latex')
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/16twomoon_loss2'],'epsc')

figure(5);clf
scatter(x1,x2,40,evspe(:,2)*sqrt(n))
colorbar
caxis([-20 20])
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/16twomoon_specnet1_evec_scatter'],'epsc')

figure(6);clf
scatter(x1,x2,40,evnn(:,2)*sqrt(n))
colorbar
caxis([-20 20])
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/16twomoon_specnet2_evec_scatter'],'epsc')

figure(7);clf
scatter(x1,x2,40,evcont(:,2)*sqrt(n))
colorbar
caxis([-20 20])
set(gca, 'fontsize', fontsize);
saveas(gcf,[fpath '/16twomoon_cont_evec_scatter'],'epsc')
end



if(0)
load('specnet1_full.mat')
loss1 = loss;
y1 = y;
load('specnet2_full.mat')
loss2 = loss;
y2 = y;
figure
subplot(221)
plot(log10(loss1)')
title('SpecNet1-full-net')
subplot(222)
plot(log10(loss2)')
title('SpecNet2-full-net')
subplot(223)
plot(y1)
title('SpecNet1-full-net')
subplot(224)
plot(y2)
title('SpecNet2-full-net')
end

if(0)
load('specnet1_full_la.mat')
loss1 = losses_spec1_full;
y1 = u;
load('specnet2_full_la.mat')
loss2 = losses_spec2_full;
loss2(loss2<0) = 0;
y2 = u;
figure
subplot(221)
plot(log10(loss1))
title('SpecNet1-full-LA')
subplot(222)
plot(log10(loss2))
title('SpecNet2-full-LA')
subplot(223)
plot(y1)
title('SpecNet1-full-LA')
subplot(224)
plot(y2)
title('SpecNet2-full-LA')
end