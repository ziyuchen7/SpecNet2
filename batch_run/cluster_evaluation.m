%clear all;close all;
data_name = 'two_moons';
batch = 4;
run_num = 10;
epoch = 300;
stepsize = '0.001';
grad_type = 'specnet2_fake_neighbor';

acc_mean = zeros(2,epoch); % first row - training set; second row - testing set;
acc_std  = zeros(2,epoch);
acc_assort1 = zeros(run_num,epoch);
acc_assort2 = zeros(run_num,epoch);
error1 = zeros(2,epoch);
error2 = zeros(2,epoch);

for i = 1:run_num
    netout = load([pwd '/' data_name '/data_01/'...
        grad_type '_unit_128_depth_2_lr' '_' stepsize '_batch_' num2str(batch,'%d')...
        '-' num2str(i-1,'%d') '.mat']);
    acc_assort1(i,:) = netout.train_loss;
    acc_assort2(i,:) = netout.test_loss;
end

acc_mean(1,:) = mean(acc_assort1,1);
acc_mean(2,:) = mean(acc_assort2,1);

for j = 1:run_num
    netout = load([pwd '/' data_name '/data_01/'...
        grad_type '_unit_128_depth_2_lr' '_' stepsize '_batch_' num2str(batch,'%d')...
        '-' num2str(j-1,'%d') '.mat']);
    idx1 = find(netout.train_loss-acc_mean(1,:)>0);
    cidx1 = setdiff(1:epoch,idx1);
    error1(1,idx1) = max(error1(1,idx1),netout.train_loss(idx1)-acc_mean(1,idx1));
    error1(2,cidx1) = max(error1(2,cidx1),acc_mean(1,cidx1)-netout.train_loss(cidx1));
    idx2 = find(netout.test_loss-acc_mean(2,:)>0);
    cidx2 = setdiff(1:epoch,idx2);
    error2(1,idx2) = max(error2(1,idx2),netout.test_loss(idx2)-acc_mean(2,idx2));
    error2(2,cidx2) = max(error2(2,cidx2),acc_mean(2,cidx2)-netout.test_loss(cidx2));
end

acc_std(1,:)  = std(acc_assort1);
acc_std(2,:)  = std(acc_assort2);

% figure()
% plot(acc_mean(1,:));
% hold on
% plot(acc_mean(2,:));
% plot(acc_std(1,:));
% plot(acc_std(2,:));
% legend('train-mean','test-mean','train-std','test-std','Location','east');
% xlabel('Epoch')
% ylim([0 1])
% set(gca,'fontsize',30)

figure()
errorbar(1:epoch,acc_mean(1,:),error1(2,:),error1(1,:));
hold on
errorbar(1:epoch,acc_mean(2,:),error2(2,:),error2(1,:));
legend('train','test','Location','east');
xlabel('Epoch')
ylim([0.5 1])
set(gca,'fontsize',20)