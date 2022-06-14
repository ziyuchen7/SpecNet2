function [num] = neighbor_check(W, batch_size)

num = 0;
iter = 30;

for i = 1:iter
    idx = randsample(size(W,1),batch_size);
    temp_num = length(find(sum(W(idx,:),1)));
    num = num + temp_num;
end

num = num/iter;



end
