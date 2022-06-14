function gen_data(opts)

switch opts.name
    case 'one_moon'
        gen_mat(opts);
    case 'two_moons'
        gen_mat(opts);
    case 'spiral'
        gen_spiral(opts);
    case 'ring'
        gen_ring(opts);
end

end