function gen_ring(opts)

n         = opts.n;
sigma     = opts.sigma;
epsx      = opts.epsilon;
kmax      = opts.num_evals;
sp_lvl    = opts.sparse_level;
plot_flag = opts.plot_flag;
path      = opts.save_path;

% map_to_RD_func = @(t) 1/(sqrt(5)*2*pi)*[...
%             cos(2*pi * t), ...
%             sin(2*pi * t), ...
%             2/omegaM*cos(2*pi * omegaM*t), ...
%             2/omegaM*sin(2*pi * omegaM*t)];

%% method 1
%theta = linspace(0,2*pi,n)';
%x = [cos(theta), sin(theta)];

%% method 2
% theta = linspace(0,1,n)';
% x = map_to_RD_func(theta);

%% method 3
map_to_RD_func = @(t) [...
(1.5 + sin(t)) .* cos(4*t)/2,...
(1.5 + sin(t)) .* sin(4*t)/2,...
cos(t)]/2;

% map_to_RD_func = @(t) [...
%     (2 + sin(t)) .* cos(4*t)/4,...
%     (2 + sin(t)) .* sin(4*t)/4,...
%     cos(t)]/2;

theta = linspace(0,2*pi,n)';
x = map_to_RD_func(theta);

x = x + randn(size(x))*epsx;

if plot_flag
    figure(1)
    scatter3(x(:,1),x(:,2),x(:,3),40,theta)
    axis equal
end
%%
W0 = gaufunc(x,x,sigma);

%% sparsify the affinity
W0 = W0.*(abs(W0) > sp_lvl);

dW0 = sum(W0,2);

W = W0./(dW0*dW0');
tmp = median(dW0).^2;
W = W*tmp;

spW = sparse(W);

if plot_flag
    figure(2)
    imagesc(W)
    
    figure(3)
    scatter3(x(:,1),x(:,2),x(:,3),40,W(:,1))
    colorbar
end
%%
[evecs,evals] = eigs(W, diag(sum(W,2)), kmax);
evals = diag(evals);
evecs = evecs*n;

if plot_flag
    figure(4)
    % plot(evecs)
    scatter(evecs(:,2),evecs(:,3),40,theta);
    
    figure(5)
    bar(1-diag(evals))
end
%%
save([path '/disc.mat'], 'n', 'x', 'W', 'spW', 'evecs', 'evals')

end