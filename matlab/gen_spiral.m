function gen_spiral(opts)

n         = opts.n;
sigma     = opts.sigma;
epsx      = opts.epsilon;
kmax      = opts.num_evals;
sp_lvl    = opts.sparse_level;
plot_flag = opts.plot_flag;
path      = opts.save_path;

map_to_RD_func = @(t) [...
    (1.5 + sin(t)) .* cos(4*t)/2,...
    (1.5 + sin(t)) .* sin(4*t)/2,...
    cos(t)]/1.5;

theta = linspace(0,pi,n)';
x = map_to_RD_func(theta);

x = x + randn(size(x))*epsx;

if(plot_flag)
    figure(1)
    scatter3(x(:,1),x(:,2),x(:,3),40,theta)
    axis equal
end

W0 = gaufunc(x,x,sigma);

W0 = W0.*(abs(W0) > sp_lvl);

dW0 = sum(W0,2);

W = W0./(dW0*dW0');
tmp = median(dW0).^2;
W = W*tmp;

spW = sparse(W);

if(plot_flag)
    figure(2)
    imagesc(W)
    
    figure(3)
    scatter3(x(:,1),x(:,2),x(:,3),40,W(:,1))
    colorbar
end

[evecs,evals] = eigs(W, diag(sum(W,2)), kmax);
evals = diag(evals);
evecs = evecs*n;

if(plot_flag)
    figure(4)
    scatter(evecs(:,2),evecs(:,3),40,theta);
    
    figure(5)
    bar(1-diag(evals))
end

save([path '/disc.mat'], 'n', 'x', 'W', 'spW', 'evecs', 'evals');

end