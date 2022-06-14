function [O] = rot_matrix(Y)
% calculate the best rotation matrix to make the first column of Y closest 
% to constant vector in l2 sense.
dim = size(Y,1);
b = ones(dim,1);
T = Y\b;
nor = sqrt(sum(T.^2));
T = T/nor;
O = [T(1),-T(2);T(2),T(1)];

end

