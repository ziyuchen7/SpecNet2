clear all; close all;

%% plot ground truth
train = load('./disc.mat');
test = load('../test.mat');
evcont1 = train.evcont;
evcont2 = test.evcont;
trainx = train.x;
testx = test.x;

figure(1)
scatter(trainx(:,1),trainx(:,2),30,evcont1(:,2),'o');
grid on; colormap(jet); 
title('First nontrivial eigenfunction (train)')
set(gca, 'fontsize', 24);

figure(2)
scatter(testx(:,1),testx(:,2),30,evcont2(:,2),'o');
grid on; colormap(jet); 
title('First nontrivial eigenfunction (test)')
set(gca, 'fontsize', 24);


%% embedding by SpecNet1
mat = load('./specnet1_fake_neighboru_unit_128_depth_2_lr_1e-05_batch_4-8');

evcont1 = -mat.y;
evcont2 = -mat.testy;

figure(1)
scatter(trainx(:,1),trainx(:,2),30,evcont1(:,2),'o');
grid on; colormap(jet); 
title({'SpecNet1-neighbor', '1st nontrivial eigenfunction (train)'})
set(gca, 'fontsize', 24);

figure(2)
scatter(testx(:,1),testx(:,2),30,evcont2(:,2),'o');
grid on; colormap(jet); 
title({'SpecNet1-neighbor', '1st nontrivial eigenfunction (test)'})
set(gca, 'fontsize', 24);


%% embedding by SpecNet2
mat = load('./specnet2_fake_neighbor_unit_128_depth_2_lr_0.001_batch_4-0.mat');

evcont1 = mat.y;
evcont2 = mat.testy;

figure(1)
scatter(trainx(:,1),trainx(:,2),30,evcont1(:,1),'o');
grid on; colormap(jet); 
title({'SpecNet2-neighbor', '1st nontrivial eigenfunction (train)'})
set(gca, 'fontsize', 24);

figure(2)
scatter(testx(:,1),testx(:,2),30,evcont2(:,1),'o');
grid on; colormap(jet); 
title({'SpecNet2-neighbor', '1st nontrivial eigenfunction (test)'})
set(gca, 'fontsize', 24);

