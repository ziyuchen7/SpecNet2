function  ss = spec2_ls(gAg,xAg,gBg,xBg,xBx)
c = zeros(4,1);
c(1) = sum(sum(gBg.^2));
c(2) = -3*sum(sum(gBg.*xBg));
c(3) = trace(-gAg) + sum(sum(gBg.*xBx + xBg.*xBg' + xBg.^2));
c(4) = trace(xAg) - sum(sum(xBg.*xBx));
rts = roots(c);
ridx = abs(imag(rts)) < 1e-12;
rrts = real(rts(ridx));
if length(rrts) == 1
    ss = rrts;
elseif length(rrts) == 3
    rrts = sort(rrts);
    if abs(rrts(3) - rrts(2)) > abs(rrts(1) - rrts(2))
        ss = rrts(3);
    else
        ss = rrts(1);
    end
else
    error('Invalid Line Search');
end
end