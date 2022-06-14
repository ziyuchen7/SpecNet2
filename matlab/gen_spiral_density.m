function [x, x1, x2, p] = gen_spiral_density(epsx)

if epsx == 0
    epsx = 0.1;
end

dtheta = 1e-3;
theta = (0:dtheta:pi)';

dx = 2e-2;
[x1,x2,x3] = meshgrid(-1.2:dx:1.2, -1.2:dx:1.2, -1:dx:1);
x = [x1(:) x2(:) x3(:)];

% p = sum(gaufunc(x, [(1.5 + sin(theta)) .* cos(4*theta)/2 (1.5 + sin(theta))...
%     .* sin(4*theta)/2 cos(theta)]/1.5, epsx),2)*dtheta;

p = zeros(size(x,1),1);

for i = 1:length(p)
    p(i) = sum(gaufunc(x(i,:), [(1.5 + sin(theta)) .* cos(4*theta)/2 (1.5 + sin(theta))...
        .* sin(4*theta)/2 cos(theta)]/1.5, epsx))*dtheta;
end

end