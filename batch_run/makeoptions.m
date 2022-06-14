function opts = makeoptions()

opts.la.power_orders= [0];
opts.la.methods = {'spec1_full', 'spec2_full'};

opts.la_repetition = 1;
opts.data_repetition = 1;

% create names for initialization
opts.data.name         = 'two_moons';
opts.data.n            = 2000;
opts.data.sigma        = 0.2;
opts.data.epsilon      = 0.1;
opts.data.num_evals    = 3;
opts.data.num_columns  = 2;
opts.data.sparse_level = 0.2;
opts.data.gap          = 0;
opts.data.plot_flag    = 0;
opts.data.save_path    = [];

end
