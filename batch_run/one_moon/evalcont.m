cont = load('./cont.mat');
disc = load('./test.mat');

n = 2000;
kmax = 3;

x1  = disc.x(:,1);
x2  = disc.x(:,2);
x1grid = cont.x1grid;
x2grid = cont.x2grid;
evcont = zeros(n, kmax);

for it = 1:kmax
  evcont(:,it) = interp2(x1grid, x2grid, ...
    reshape(cont.evecs(:,it),size(x1grid)), x1, x2, 'spline');
end
