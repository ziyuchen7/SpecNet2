function [batchidx, numbatch] = gen_batch(N, bs)
numbatch = ceil(N/bs);
fullidx = randperm(N);
batchidx = cell(numbatch,1);
for it = 1:numbatch
    if it == numbatch
        batchidx{it} = fullidx(bs*(it-1)+1:end);
    else
        batchidx{it} = fullidx(bs*(it-1)+(1:bs));
    end
end
end
