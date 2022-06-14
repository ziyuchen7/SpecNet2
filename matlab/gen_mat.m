function gen_mat(opts)

n          = opts.n;
gapx       = opts.gap;
epsx       = opts.epsilon;
sig_kernel = opts.sigma;
kmax       = opts.num_evals;
sp_lvl     = opts.sparse_level;
path       = opts.save_path;

switch opts.name
    case 'two_moons'
        [x, ~, ~] = gen_two_moons(n,epsx,gapx);
        W = gaufunc(x,x,sig_kernel);
        W = W.*(abs(W) > sp_lvl);
        W = (W+W')/2;
        spW = sparse(W);
        D = sum(W,2);
        [evecs, evals] = eigs(W, diag(D), kmax);
        evecs = evecs*n;
        evals = diag(evals);
        label = zeros(size(x,1),1);
        label(1:n/2) = 1;
        label(n/2+1:end) = 2;
        save([path '/disc.mat'], 'n', 'x', 'evecs', 'W', 'spW', 'evals', 'sig_kernel', 'label')
        
    case 'one_moon'
        [x, ~, ~] = gen_one_moon(n,epsx);
        W = gaufunc(x,x,sig_kernel);
        % sparsify W
        W = W.*(abs(W) > sp_lvl);
        W = (W+W')/2;
        spW = sparse(W);
        D = sum(W,2);
        [evecs, evals] = eigs(W, diag(D), kmax);
        evecs = evecs*n;
        evals = diag(evals);
        save([path '/disc.mat'], 'n', 'x', 'evecs', 'W', 'spW', 'evals', 'sig_kernel')
        
end

end
