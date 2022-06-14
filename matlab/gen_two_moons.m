 function [x, x1, x2] = gen_two_moons(n,epsx,gapx)

tt1 = rand(n/2,1);
x1 = [cos(tt1*pi),sin(tt1*pi)];
x1(:,1) = x1(:,1)-.5;
x1(:,2) = x1(:,2)-gapx;

tt2 = rand(n/2,1);
x2 = [cos(tt2*pi),sin(tt2*pi)];
x2 = -x2;
x2(:,1) = x2(:,1)+.5;
x2(:,2) = x2(:,2)+gapx;

x = cat(1,x1,x2);

x = x + randn(size(x))*epsx;

% x1 and x2 denote the first and second dimension here
x1 = x(:,1);
x2 = x(:,2);

end