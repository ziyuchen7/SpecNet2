function [x, x1, x2, p] = gen_ring_density(epsx)

if epsx == 0
    epsx = 0.1;
end

dtheta = 1e-3;
theta = (0:dtheta:2*pi)';

dx = 2e-2;
[x1,x2,x3] = meshgrid(-0.8:dx:0.8, -0.8:dx:0.8, -0.6:dx:0.6);
x = [x1(:) x2(:) x3(:)];

p = sum(gaufunc(x, [(1.5 + sin(theta)) .* cos(4*theta)/2 (1.5 + sin(theta))...
    .* sin(4*theta)/2 cos(theta)]/2, epsx),2)*dtheta;

end