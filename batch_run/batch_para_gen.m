% batch run
clear all; close all;
addpath('../matlab');
addpath('../matlab/LinearAlg');

opts = makeoptions();
opts.data.name = 'one_moon';
opts.folder_path = [pwd '/' opts.data.name];
save_path = opts.folder_path;

opts.la_repetition = 1;
opts.data_repetition = 1;

switch opts.data.name
    case 'spiral'
        opts.data.n            = 2000;
        opts.data.sigma        = 0.2;
        opts.data.epsilon      = 0.05;
        opts.data.num_evals    = 4;
        opts.data.num_columns  = 3;
        opts.data.sparse_level = 0.1;
    case 'ring'
        opts.data.n            = 2000;
        opts.data.sigma        = 0.2;
        opts.data.epsilon      = 0;
        opts.data.num_evals    = 4;
        opts.data.num_columns  = 3;
        opts.data.sparse_level = 0.1;
    case 'one_moon'
        opts.data.n            = 2000;
        opts.data.sigma        = 0.2;
        opts.data.epsilon      = 0.1;
        opts.data.num_evals    = 4;
        opts.data.num_columns  = 3;
        opts.data.sparse_level = 0.6;
    case 'two_moons'
        opts.data.n            = 2000;
        opts.data.sigma        = 0.15;
        opts.data.epsilon      = 0.06;
        opts.data.num_evals    = 4;
        opts.data.num_columns  = 3;
        opts.data.sparse_level = 0.08;
        opts.data.gap          = 0.3;
end

opts.data.plot_flag    = 0;
opts.data.save_path    = save_path;
if not(exist(save_path,'dir'))
    mkdir(save_path)
end
jsonstr = jsonencode(opts);
fileID = fopen([save_path '/para1.json'],'w');
fprintf(fileID,'%12s',jsonstr);
fclose(fileID);