gapx = 0.3;
epsx = 0.1;
sig_kernel = 0.2;
kmax = 2;
knn = 100;
n = 2000;

if(0)
    [x, ~, ~] = gen_two_moons(n,epsx,gapx);
    [A, B, y, evals, I, J, V] = spectral_disc_knn(x, sig_kernel, kmax, knn);
    
    path = '../results/two_moons';
    if not(exist(path,'dir'))
        mkdir(path)
    end
    save([path '/disc_knn.mat'], 'n', 'x', 'y', 'A', 'B', 'I', 'J', 'V', 'knn')
    
else
    [x, ~, ~] = gen_one_moon(n,epsx);
    [A, B, y, evals, I, J, V] = spectral_disc_knn(x, sig_kernel, kmax, knn);
    
    path = '../results/one_moon';
    if not(exist(path,'dir'))
        mkdir(path)
    end
    save([path '/disc_knn.mat'], 'n', 'x', 'y', 'A', 'B', 'I', 'J', 'V', 'knn')
    
end