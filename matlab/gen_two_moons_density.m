function [x, x1, x2, p] = gen_two_moons_density(epsx,gapx)

dtheta = 1e-3;
theta = (0:dtheta:pi)';

dx = 2e-2;
[x1,x2] = meshgrid(-2:dx:2, -1.5:dx:1.5);
x = [x1(:) x2(:)];

p = sum(gaufunc(x, [cos(theta)-0.5 sin(theta)-gapx], epsx),2)*dtheta ...
    + sum(gaufunc(x, [-cos(theta)+0.5 -sin(theta)+gapx], epsx),2)*dtheta;

end