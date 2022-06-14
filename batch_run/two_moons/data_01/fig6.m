clear all;close all;
data_name = 'two_moons';
batch = 4;
run_num = 10;
epoch = 300;

%% specnet1
stepsize = '1e-05';
grad_type = 'specnet1_fake_neighboru';

acc_mean = zeros(2,epoch); % first row - training set; second row - testing set;
acc_std  = zeros(2,epoch);
acc_assort1 = zeros(run_num,epoch);
acc_assort2 = zeros(run_num,epoch);
error1 = zeros(2,epoch);
error2 = zeros(2,epoch);

for i = 1:run_num
    netout = load([grad_type '_unit_128_depth_2_lr' '_' stepsize '_batch_' num2str(batch,'%d')...
        '-' num2str(i-1,'%d') '.mat']);
    acc_assort1(i,:) = netout.train_loss;
    acc_assort2(i,:) = netout.test_loss;
end

acc_mean(1,:) = mean(acc_assort1,1);
acc_mean(2,:) = mean(acc_assort2,1);

for j = 1:run_num
    netout = load([grad_type '_unit_128_depth_2_lr' '_' stepsize '_batch_' num2str(batch,'%d')...
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

Error1 = zeros(size(error1));
Error2 = zeros(size(error2));
Error1(1:5:end) = error1(1:5:end);
Error2(1:5:end) = error2(1:5:end);

figure(1)
errorbar(1:epoch,acc_mean(1,:),Error1(2,:),Error1(1,:));
figure(2)
errorbar(1:epoch,acc_mean(2,:),Error2(2,:),Error2(1,:));

%% specnet2
stepsize = '0.001';
grad_type = 'specnet2_fake_neighbor';

acc_mean = zeros(2,epoch); % first row - training set; second row - testing set;
acc_std  = zeros(2,epoch);
acc_assort1 = zeros(run_num,epoch);
acc_assort2 = zeros(run_num,epoch);
error1 = zeros(2,epoch);
error2 = zeros(2,epoch);

for i = 1:run_num
    netout = load([grad_type '_unit_128_depth_2_lr' '_' stepsize '_batch_' num2str(batch,'%d')...
        '-' num2str(i-1,'%d') '.mat']);
    acc_assort1(i,:) = netout.train_loss;
    acc_assort2(i,:) = netout.test_loss;
end

acc_mean(1,:) = mean(acc_assort1,1);
acc_mean(2,:) = mean(acc_assort2,1);

for j = 1:run_num
    netout = load([grad_type '_unit_128_depth_2_lr' '_' stepsize '_batch_' num2str(batch,'%d')...
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

Error1 = zeros(size(error1));
Error2 = zeros(size(error2));
Error1(1:5:end) = error1(1:5:end);
Error2(1:5:end) = error2(1:5:end);

figure(1)
hold on
errorbar(1:epoch,acc_mean(1,:),Error1(2,:),Error1(1,:));
xlabel('Epoch');ylabel('Accuracy')
ylim([0.5 1])
legend('SpecNet1-Neighbor','SpecNet2-Neighbor','Location','southeast')
title('Classification accuracy on training data')
set(gca,'fontsize',24)

figure(2)
hold on
errorbar(1:epoch,acc_mean(2,:),Error2(2,:),Error2(1,:));
xlabel('Epoch');ylabel('Accuracy')
ylim([0.5 1])
legend('SpecNet1-Neighbor','SpecNet2-Neighbor','Location','southeast')
title('Classification accuracy on testing data')
set(gca,'fontsize',24)