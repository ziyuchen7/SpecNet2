function [x, x1, x2, p] = gen_one_moon_density(epsx)

dtheta = 1e-3;
theta = (0:dtheta:pi)';

dx = 2e-2;
[x1,x2] = meshgrid(-2:dx:2, -1.5:dx:1.5);
x = [x1(:) x2(:)];

p = sum(gaufunc(x, [cos(theta) sin(theta)], epsx),2)*dtheta;

end