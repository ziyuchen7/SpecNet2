clear all; close all;


net1_full = load(['./LA/continuous/spec1_full-la.mat']);

net1_local = load(['./LA/continuous/spec1_local-la.mat']);

net1_neighbor = load(['./LA/continuous/spec1_neighbor-la.mat']);

net2_full = load(['./LA/continuous/spec2_full-la.mat']);

net2_local = load(['./LA/continuous/spec2_local-la.mat']);

net2_neighbor = load(['./LA/continuous/spec2_neighbor-la.mat']);

figure(1) % plot relative errors for first nontrivial eigenvector
plot(log10(net1_full.error(:,2)),'color','#0072BD','LineStyle','-','LineWidth',2);
hold on
plot(log10(net1_local.error(:,2)),'color','#0072BD','LineStyle','--','LineWidth',2);
plot(log10(net1_neighbor.error(:,2)),'color','#0072BD','LineStyle',':','LineWidth',2);

plot(log10(net2_full.error(:,1)),'color','#D95319','LineStyle','-','LineWidth',2);
plot(log10(net2_local.error(1:300,1)),'color','#D95319','LineStyle','--','LineWidth',2);
plot(log10(net2_neighbor.error(:,1)),'color','#D95319','LineStyle',':','LineWidth',2);
xlabel('Epoch');ylabel('$\log_{10}$','Interpreter','latex')
ylim([-3 0.5])
legend('SpecNet1-full','SpecNet1-local','SpecNet1-neighbor','SpecNet2-full',...
    'SpecNet2-local','SpecNet2-neighbor','Location','southwest')
title('First nontrivial eigenfunction')
set(gca,'fontsize',24)

figure(2) % plot relative errors for first nontrivial eigenvector
plot(log10(net1_full.error(:,3)),'color','#0072BD','LineStyle','-','LineWidth',2);
hold on
plot(log10(net1_local.error(:,3)),'color','#0072BD','LineStyle','--','LineWidth',2);
plot(log10(net1_neighbor.error(:,3)),'color','#0072BD','LineStyle',':','LineWidth',2);

plot(log10(net2_full.error(:,2)),'color','#D95319','LineStyle','-','LineWidth',2);
plot(log10(net2_local.error(1:300,2)),'color','#D95319','LineStyle','--','LineWidth',2);
plot(log10(net2_neighbor.error(:,2)),'color','#D95319','LineStyle',':','LineWidth',2);
xlabel('Epoch');ylabel('$\log_{10}$','Interpreter','latex')
ylim([-3 0.5])
legend('SpecNet1-full','SpecNet1-local','SpecNet1-neighbor','SpecNet2-full',...
    'SpecNet2-local','SpecNet2-neighbor','Location','southwest')
title('Second nontrivial eigenfunction')
set(gca,'fontsize',24)