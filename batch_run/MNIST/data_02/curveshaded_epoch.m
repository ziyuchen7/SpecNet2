% clear; close all;
c = zeros(6,1);
d = zeros(6,1);
epoch = zeros(6,1);
batch = zeros(6,1);
c(1) = 3.3; c(2) = 2.3; c(3) = 2; c(4) = 28.2; c(5) = 15.6; c(6) = 9.6;
epoch(1) = 726; epoch(2) = 1044; epoch(3) = 1200; epoch(4) = 300; epoch(5) = 154; epoch(6) = 250;
batch(1) = 28; batch(2) = 56; batch(3) = 112; batch(4) = 2; batch(5) = 4; batch(6) = 8;

color = [ 0 0.4470 0.7410;
          0.8500 0.3250 0.0980;
          0.9290 0.6940 0.1250;
          0.4940 0.1840 0.5560;
          0.4660 0.6740 0.1880;
          0.3010 0.7450 0.9330];

for k = 1:6
    acc_mean = zeros(2,epoch(k));
    acc_std  = zeros(2,epoch(k));
    acc_assort1 = zeros(10,epoch(k));
    acc_assort2 = zeros(10,epoch(k));
    
    if k <= 3
        grad_type = 'specnet1_localu';
    else
        grad_type = 'specnet2_neighbor';
    end

    for i = 1:10
        if k == 4
            netout = load([grad_type '_unit_256_depth_3_lr_0.0001' '_batch_' num2str(batch(k),'%d')...
            '-' num2str(i+9,'%d') '.mat']);
        else
            netout = load([grad_type '_unit_256_depth_3_lr_0.0001' '_batch_' num2str(batch(k),'%d')...
            '-' num2str(i-1,'%d') '.mat']);
        end
        acc_assort1(i,:) = netout.grad_loss(1,:);
        acc_assort2(i,:) = netout.grad_loss(2,:);
    end
    
    acc_mean(1,:) = mean(acc_assort1,1);
    acc_mean(2,:) = mean(acc_assort2,1);
    acc_std(1,:)  = std(acc_assort1);
    acc_std(2,:)  = std(acc_assort2);
    
    x = 1:epoch(k);
    x2 = [x, fliplr(x)];
    inBetween1 = [log10(acc_mean(1,:) + acc_std(1,:)), log10(fliplr(acc_mean(1,:) - acc_std(1,:)))];
    inBetween2 = [log10(acc_mean(2,:) + acc_std(2,:)), log10(fliplr(acc_mean(2,:) - acc_std(2,:)))];
%     inBetween1 = [acc_mean(1,:) + acc_std(1,:), fliplr(acc_mean(1,:) - acc_std(1,:))];
%     inBetween2 = [acc_mean(2,:) + acc_std(2,:), fliplr(acc_mean(2,:) - acc_std(2,:))];
    figure(1)
    fill(x2, inBetween1, color(k,:),'FaceAlpha', 0.3,'linestyle','none');
    hold on
    plot(x, log10(acc_mean(1,:)), 'color', color(k,:), 'LineWidth', 2);
%     plot(x, acc_mean(1,:), 'color', color(k,:), 'LineWidth', 2);

    figure(2)
    fill(x2, inBetween2, color(k,:),'FaceAlpha', 0.3,'linestyle','none');
    hold on
    plot(x, log10(acc_mean(2,:)), 'color', color(k,:), 'LineWidth', 2);
%     plot(x, acc_mean(2,:), 'color', color(k,:), 'LineWidth', 2);

end
figure(1)
legend('','SpecNet1-local bs=28','','SpecNet1-local bs=56','','SpecNet1-local bs=112',...
    '','SpecNet2-neighbor bs=2','','SpecNet2-neighbor bs=4','','SpecNet2-neighbor bs=8')
set(gca,'fontsize',24)
xlabel('Epoch')
ylabel('$\log_{10}$','Interpreter','latex')
xlim([0 150])
ylim([-4 1])
title('Loss $f_1$','Interpreter','latex')

figure(2)
legend('','SpecNet1-local bs=28','','SpecNet1-local bs=56','','SpecNet1-local bs=112',...
    '','SpecNet2-neighbor bs=2','','SpecNet2-neighbor bs=4','','SpecNet2-neighbor bs=8')
set(gca,'fontsize',24)
xlabel('Epoch')
ylabel('$\log_{10}$','Interpreter','latex')
xlim([0 150])
ylim([-4 1])
title('Loss $f_2$','Interpreter','latex')
