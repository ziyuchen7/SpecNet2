%%
clear all;close all;
addpath('../matlab');
data_name = 'two_moons';
para_path = [pwd '/' data_name '/para.json'];

jsonstr = fileread(para_path);
options = jsondecode(jsonstr);
opts    = options.data;
data_index = 1;
power = 0;
la_method = 'spec2_full';
cont = load([pwd '/' data_name '/cont.mat']);
n    = opts.n;
kmax = opts.num_columns;


%% evaluate on each data realization for 3d data
evec_error = zeros(5,3);
eval_error = zeros(5,3);
disc = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') '/disc.mat']);
evdisc = disc.evecs;
D = diag(sum(disc.W,2));
x1  = disc.x(:,1);
x2  = disc.x(:,2); 
x3  = disc.x(:,3);
x1grid = cont.x1grid;
x2grid = cont.x2grid;
x3grid = cont.x3grid;
evcont = zeros(n, kmax);

for it = 1:kmax
    evcont(:,it) = interp3(x1grid, x2grid, x3grid, ...
        reshape(cont.evecs(:,it),size(x1grid)), x1, x2, x3, 'spline');

end
    
for it_la = 1:5
    la = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') '/'...
        la_method '_' num2str(power,'%02d') '_' num2str(it_la,'%02d') '/la.mat']);
    evla = la.u;
    for kk = 1:kmax
    evec_error(it_la,kk) = norm(evla(:,kk)-sign(evla(:,kk)'*evcont(:,kk))*evcont(:,kk))/norm(evcont(:,kk));
    eval_error(it_la,kk) = abs(la.evals(kk) - cont.evals(kk))/abs(1-cont.evals(kk));
    end
    
end

sum(evec_error,1)/1
sum(eval_error,1)/1

%% evaluate on each data realization for 2d data
evec_error = zeros(5,3);
eval_error = zeros(5,3);
disc_error = zeros(1,2);
disc = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') '/disc.mat']);
evdisc = disc.evecs;
D = diag(sum(disc.W,2));
x1  = disc.x(:,1);
x2  = disc.x(:,2);
x1grid = cont.x1grid;
x2grid = cont.x2grid;
evcont = zeros(n, kmax);

for it = 1:kmax
      evcont(:,it) = interp2(x1grid, x2grid, ...
        reshape(cont.evecs(:,it),size(x1grid)), x1, x2, 'spline');
end

evcont = evcont./sqrt(diag(evcont'*D*evcont)')*disc.n;


for it_la = 1:1
    la = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') '/'...
         'batch_4/' la_method '_' num2str(power,'%02d') '_' num2str(it_la,'%02d') '/la.mat']);
    evla = la.u;
    if strcmp(data_name, 'two_moons')
        O = rot_matrix(evdisc(:,1:2));
        evdisc(:,1:2) = evdisc(:,1:2)*O;
        Ola = rot_matrix(evla(:,1:2));
        evla(:,1:2) = evla(:,1:2)*Ola;
    end
    for kk = 1:kmax
    evec_error(it_la,kk) = norm(evla(:,kk)-sign(evla(:,kk)'*evcont(:,kk))*evcont(:,kk))/norm(evcont(:,kk));
    eval_error(it_la,kk) = abs(la.evals(kk) - cont.evals(kk))/abs(1-cont.evals(kk));
    disc_error(it_la,kk) = norm(evdisc(:,kk)-sign(evdisc(:,kk)'*evcont(:,kk))*evcont(:,kk))/norm(evcont(:,kk));
    end
    
end

log10(sum(evec_error,1)/1)
log10(sum(eval_error,1)/1)
log10(sum(disc_error,1)/1)
% save([pwd '/' data_name '/test.mat'], 'x', 'evcont')
%%
figure()
plot(log10(abs(la.loss)))
title([num2str(log10(sum(evec_error(:,1),1)/1)) ' , ' num2str(log10(sum(evec_error(:,2),1)/1))...
    ' and ' num2str(la.evals(1)) ' , ' num2str(la.evals(2))])


%% run kmeans classification
disc = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') '/disc.mat']);
la = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') '/'...
          la_method '_00' '_01' '/la.mat']);
evla = la.u;
evdisc = disc.evecs;
evdisc = evdisc(:,1:2);

idx0 = kmeans(evdisc,2);
idx1 = kmeans(evla,2);

score0 = mean(abs(idx0 - [2*ones(n/2,1);ones(n/2,1)]));
score0 = max(score0, 1-score0)
score1 = mean(abs(idx1 - [2*ones(n/2,1);ones(n/2,1)]));
score1 = max(score1, 1-score1)
%% network evaluation
net = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') '/'...
        'batch_4' '/specnet2_full_unit_128_depth_2_lr_0.001.mat']);
evnet = net.y;
net_error = zeros(1,2);
if strcmp(data_name, 'two_moons')
        Onet = rot_matrix(evnet(:,1:2));
        evnet(:,1:2) = evnet(:,1:2)*Onet;
end

for kk = 1:kmax
    net_error(it_la,kk) = norm(evnet(:,kk)-sign(evnet(:,kk)'*evcont(:,kk))*evcont(:,kk))/norm(evcont(:,kk));
end

figure()
plot(log10(abs(net.train_loss')))
title(['net: ' num2str(log10(sum(net_error,1)/1)) ' and ' num2str(net.evals)])

figure()
plot(log10(abs(net.train_loss')))
hold on
plot(log10(abs(net.test_loss')))
legend('1st-train','2nd-train','1st-test','2nd-test')
title('eigenfunction loss')


%% network evaluation
data_name = 'two_moons';
data_index = 7;
net = load([pwd '/' data_name '/data_' num2str(data_index,'%02d') ...
         '/specnet1_full_unit_128_depth_2_lr_0.0001.mat']);

figure()
plot(log10(net.grad_loss'))

figure()
plot(net.train_loss)
hold on
plot(net.test_loss)
legend('train','test')