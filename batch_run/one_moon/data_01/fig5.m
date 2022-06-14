clear all; close all;

batch = 4;
nbr = 620;
N = 2000;
epoch = 300;
stepsize1 = '0.0001';
stepsize2 = '0.001';

net1_full = load(['specnet1_fullu' '_unit_128_depth_2_lr' '_' stepsize1...
                      '_batch_' num2str(batch,'%d') '-0' '.mat']);

net1_local = load(['specnet1_localu' '_unit_128_depth_2_lr' '_' stepsize1...
                      '_batch_' num2str(batch,'%d') '-0' '.mat']);

net1_neighbor = load(['specnet1_fake_neighboru' '_unit_128_depth_2_lr' '_' stepsize1...
                      '_batch_' num2str(batch,'%d') '-0' '.mat']);

net2_full = load(['specnet2_full' '_unit_128_depth_2_lr' '_' stepsize2...
                      '_batch_' num2str(batch,'%d') '-0' '.mat']);

net2_local = load(['specnet2_local' '_unit_128_depth_2_lr' '_' stepsize2...
                      '_batch_' num2str(batch,'%d') '-0' '.mat']);

net2_neighbor = load(['specnet2_neighbor' '_unit_128_depth_2_lr' '_' stepsize2...
                      '_batch_' num2str(batch,'%d') '-0' '.mat']);

scale1 = N/batch*N*(1:epoch); % for full update 
scale2 = N/batch*batch*(1:epoch); % for local update
scale3 = N/batch*nbr*(1:1000); % for neighbor update

figure(1) % plot relative errors for first nontrivial eigenvector
plot(scale1,log10(net1_full.train_loss(2,:)),'color','#0072BD','LineStyle','-','LineWidth',2);
hold on
plot(scale3,log10(net1_neighbor.train_loss(2,:)),'color','#0072BD','LineStyle',':','LineWidth',2);
plot(scale1,log10(net2_full.train_loss(1,:)),'color','#D95319','LineStyle','-','LineWidth',2);
plot(scale3,log10(net2_neighbor.train_loss(1,:)),'color','#D95319','LineStyle',':','LineWidth',2);
xlabel('Computational cost');ylabel('$\log_{10}$','Interpreter','latex')
xlim([0 2e8])
ylim([-1.5 0])
legend('SpecNet1-full','SpecNet1-neighbor','SpecNet2-full',...
    'SpecNet2-neighbor')
title({'Relative error', '1st nontrivial eigenfunction'})
set(gca,'fontsize',24)

figure(2) % plot relative errors for first nontrivial eigenvector
plot(scale1,log10(net1_full.train_loss(3,:)),'color','#0072BD','LineStyle','-','LineWidth',2);
hold on
plot(scale3,log10(net1_neighbor.train_loss(3,:)),'color','#0072BD','LineStyle',':','LineWidth',2);
plot(scale1,log10(net2_full.train_loss(2,:)),'color','#D95319','LineStyle','-','LineWidth',2);
plot(scale3,log10(net2_neighbor.train_loss(2,:)),'color','#D95319','LineStyle',':','LineWidth',2);
xlabel('Computational cost');ylabel('$\log_{10}$','Interpreter','latex')
xlim([0 2e8])
ylim([-1.5 0])
legend('SpecNet1-full','SpecNet1-neighbor','SpecNet2-full','SpecNet2-neighbor')
title({'Relative error', '2nd nontrivial eigenfunction'})
set(gca,'fontsize',24)