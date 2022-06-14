clear all; close all;

stepsize1 = '0.0001';
stepsize2 = '0.001';
batch = 4;

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

net2_neighbor = load(['specnet2_fake_neighbor' '_unit_128_depth_2_lr' '_' stepsize2...
                      '_batch_' num2str(batch,'%d') '-0' '.mat']);

figure(1) % plot relative errors for first nontrivial eigenvector
plot(log10(net1_full.test_loss(2,:)),'color','#0072BD','LineStyle','-','LineWidth',2);
hold on
plot(log10(net1_local.test_loss(2,:)),'color','#0072BD','LineStyle','--','LineWidth',2);
plot(log10(net1_neighbor.test_loss(2,1:300)),'color','#0072BD','LineStyle',':','LineWidth',2);

plot(log10(net2_full.test_loss(1,:)),'color','#D95319','LineStyle','-','LineWidth',2);
plot(log10(net2_local.test_loss(1,:)),'color','#D95319','LineStyle','--','LineWidth',2);
plot(log10(net2_neighbor.test_loss(1,1:300)),'color','#D95319','LineStyle',':','LineWidth',2);
xlabel('Epoch');ylabel('$\log_{10}$','Interpreter','latex')
ylim([-1.5 0]);
legend('SpecNet1-full','SpecNet1-local','SpecNet1-neighbor','SpecNet2-full',...
    'SpecNet2-local','SpecNet2-neighbor')
title('First nontrivial eigenfunction (testing)')
set(gca,'fontsize',24)

figure(2) % plot relative errors for first nontrivial eigenvector
plot(log10(net1_full.test_loss(3,:)),'color','#0072BD','LineStyle','-','LineWidth',2);
hold on
plot(log10(net1_local.test_loss(3,:)),'color','#0072BD','LineStyle','--','LineWidth',2);
plot(log10(net1_neighbor.test_loss(3,1:300)),'color','#0072BD','LineStyle',':','LineWidth',2);

plot(log10(net2_full.test_loss(2,:)),'color','#D95319','LineStyle','-','LineWidth',2);
plot(log10(net2_local.test_loss(2,:)),'color','#D95319','LineStyle','--','LineWidth',2);
plot(log10(net2_neighbor.test_loss(2,1:300)),'color','#D95319','LineStyle',':','LineWidth',2);
xlabel('Epoch');ylabel('$\log_{10}$','Interpreter','latex')
ylim([-1.5 0]);
legend('SpecNet1-full','SpecNet1-local','SpecNet1-neighbor','SpecNet2-full',...
    'SpecNet2-local','SpecNet2-neighbor')
title('Second nontrivial eigenfunction (testing)')
set(gca,'fontsize',24)