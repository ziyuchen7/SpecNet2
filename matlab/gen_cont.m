function gen_cont(opts)


sig_kernel = opts.sigma;
epsx       = opts.epsilon;
kmax       = opts.num_columns;
gapx       = opts.gap;

switch opts.name
    case 'two_moons'
    [x, x1grid, x2grid, p]   = gen_two_moons_density(epsx,gapx);
    [evecs, evals] = spectral_cont(x,p,sig_kernel,kmax);
    evals = diag(evals);
    save([opts.save_path '/cont.mat'], ...
        'gapx', 'epsx', 'sig_kernel', 'kmax', ...
        'evecs', 'evals', 'x1grid', 'x2grid')
    
    
    case 'one_moon'
    [x, x1grid, x2grid, p]   = gen_one_moon_density(epsx);
    [evecs, evals] = spectral_cont(x,p,sig_kernel,kmax);
    evals = diag(evals);
    save([opts.save_path '/cont.mat'], ...
        'gapx', 'epsx', 'sig_kernel', 'kmax', ...
        'evecs', 'evals', 'x1grid', 'x2grid')
    
    case 'spiral'
    [x, x1grid, x2grid, p]   = gen_spiral_density(epsx);
    [evecs, evals] = spectral_cont(x,p,sig_kernel,kmax);
    evals = diag(evals);
    save([opts.save_path '/cont.mat'], ...
        'gapx', 'epsx', 'sig_kernel', 'kmax', ...
        'evecs', 'evals', 'x1grid', 'x2grid')
    
    case 'ring'
    [x, x1grid, x2grid, p]   = gen_ring_density(epsx);
    [evecs, evals] = spectral_cont(x,p,sig_kernel,kmax);
    evals = diag(evals);
    save([opts.save_path '/cont.mat'], ...
        'gapx', 'epsx', 'sig_kernel', 'kmax', ...
        'evecs', 'evals', 'x1grid', 'x2grid')
end

end