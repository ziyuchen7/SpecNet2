data_type = 'sparse_one_moon/';

disc = load(['../results/',data_type,'disc.mat']);
cont = load(['../results/',data_type,'cont.mat']);
evecs = disc.evecs;
evals = diag(cont.evals);

rotflag = 0;
n  = disc.n;
kmax = cont.kmax-1;

x1  = disc.x(:,1);
x2  = disc.x(:,2);
x1grid = cont.x1grid;
x2grid = cont.x2grid;
evcont = zeros(n, kmax);

for it = 1:kmax
  evcont(:,it) = interp2(x1grid, x2grid, ...
    reshape(cont.evecs(:,it),size(x1grid)), x1, x2, 'spline');
end
evcont = evcont./sqrt(diag(evcont'*diag(sum(disc.W,2))*evcont))'*n;

% file1 = load('spe_net.mat');
folder_path = ['../results/',data_type,'batch_128/'];
folder_path2 = ['../results/',data_type];

fn1 = 'specnet1_full_unit_128_depth_2_lr_0.1';
fn2 = 'specnet2_full_unit_128_depth_2_lr_0.1';
% fn3 = 'specnet1_fake_neighbor_unit_128_depth_2_lr_0.001';
% fn4 = 'specnet2_fake_neighbor_unit_128_depth_2_lr_0.001';
fn3 = 'specnet1_full_power_unit_128_depth_2_lr_0.001';
fn4 = 'specnet2_full_power_unit_128_depth_2_lr_0.1';
fn5 = 'specnet1_full_la.mat';
fn6 = 'specnet2_full_la.mat';
fn7 = 'specnet1_full_power_la.mat';
fn8 = 'specnet2_full_power_la.mat';

file1 = load([folder_path,fn1,'.mat']);
file2 = load([folder_path,fn2,'.mat']);
file3 = load([folder_path,fn3,'.mat']);
file4 = load([folder_path,fn4,'.mat']);

file5 = load([folder_path2,fn5]);
file6 = load([folder_path2,fn6]);
file7 = load([folder_path2,fn7]);
file8 = load([folder_path2,fn8]);


%% figure 1
figure(1)
subplot(2,2,[1,2])
plot(log10(file1.loss)')
title(fn1,'Interpreter','none')

v = file1.y;

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err1 = zeros(kmax,1);
for i = 1:kmax
    err1(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err1))

subplot(224)
%title(num2str(abs(file1.evals(2:end)'-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(file1.evals))

%% figure 2
figure(2)
subplot(2,2,[1,2])
plot(log10(file2.loss)')
title(fn2,'Interpreter','none')

v = file2.y;

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err2 = zeros(kmax,1);
for i = 1:kmax
    err2(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err2))

subplot(224)
%title(num2str(abs(file2.evals(2:end)'-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(file2.evals))

%% figure 3
figure(3)
subplot(2,2,[1,2])
plot(log10(file3.loss)')
title(fn3,'Interpreter','none')

v = file3.y;

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err3 = zeros(kmax,1);
for i = 1:kmax
    err3(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err3))

subplot(224)
%title(num2str(abs(file3.evals(2:end)'-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(file3.evals))

%% figure 4
figure(4)
subplot(2,2,[1,2])
plot(log10(file4.loss)')
title(fn4,'Interpreter','none')

v = file4.y;

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err4 = zeros(kmax,1);
for i = 1:kmax
    err4(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err4))

subplot(224)
%title(num2str(abs(file4.evals(2:end)'-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(file4.evals))

%% activate next line if need figure 5 to 9
if (1)

%% figure 5
figure(5)
subplot(2,2,[1,2])
plot(log10(abs(file5.losses)))
title(fn5,'Interpreter','none')


[eval_sorted,idx] = sort(file5.eval,'descend');
v = file5.u(:,idx);

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err5 = zeros(kmax,1);
for i = 1:kmax
    err5(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err5))

subplot(224)
%title(num2str(abs(eval_sorted(2:end)-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(eval_sorted))

%% figure 6
figure(6)
subplot(2,2,[1,2])
plot(log10(abs(file6.losses)))
title(fn6,'Interpreter','none')


[eval_sorted,idx] = sort(file6.eval,'descend');
v = file6.u(:,idx);

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err6 = zeros(kmax,1);
for i = 1:kmax
    err6(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err6))

subplot(224)
%title(num2str(abs(eval_sorted(2:end)-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(eval_sorted))


%% figure 7
figure(7)
subplot(2,2,[1,2])
plot(log10(abs(file7.losses)))
title(fn7,'Interpreter','none')


[eval_sorted,idx] = sort(file7.eval,'descend');
v = file7.u(:,idx);

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err7 = zeros(kmax,1);
for i = 1:kmax
    err7(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err7))

subplot(224)
%title(num2str(abs(eval_sorted(2:end)-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(eval_sorted))


%% figure 8
figure(8)
subplot(2,2,[1,2])
plot(log10(abs(file8.losses)))
title(fn8,'Interpreter','none')


[eval_sorted,idx] = sort(file8.eval,'descend');
v = file8.u(:,idx);

if rotflag == 1
    [O] = rot_matrix(v);
    v = v * O;
end

err8 = zeros(kmax,1);
for i = 1:kmax
    err8(i) = norm(v(:,i)-sign(v(:,i)'*evcont(:,i))*...
    evcont(:,i))/norm(evcont(:,i));
end

subplot(223)
plot(v)
title(num2str(err8))

subplot(224)
%title(num2str(abs(eval_sorted(2:end)-evals(2:kmax))./abs(1-evals(2:kmax))))
title(num2str(eval_sorted))

%% figure 9
figure(9)

% subplot(121)
% scatter(disc.evecs(:,2),disc.evecs(:,3),40);

subplot(122)
plot(disc.evecs(:,1:kmax))
title(num2str(diag(disc.evals(1:kmax,1:kmax))))

end