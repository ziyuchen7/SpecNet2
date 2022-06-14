function [x, x1, x2] = gen_one_moon(n, epsx)

tt = rand(n,1);
x = [cos(tt*pi),sin(tt*pi)];

x = x + randn(size(x))*epsx;

% x1 and x2 denote the first and second dimension here
x1 = x(:,1);
x2 = x(:,2);

end