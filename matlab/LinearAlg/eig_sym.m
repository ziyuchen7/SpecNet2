function [V,D] = eig_sym(A,B)
A = (A'+A)/2;
B = (B'+B)/2;
[V,D] = eig(A,B,'chol');
V = V./sqrt(diag(V'*B*V));
end