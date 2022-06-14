para_path = 'one_moon/para.json';
jsonstr = fileread(para_path);
options = jsondecode(jsonstr);

gen_cont(options.data)
for data_rep_it = 1:options.data_repetition
    save_path = [options.folder_path '/data_' num2str(data_rep_it,'%02d')];
    if not(exist(save_path,'dir'))
        mkdir(save_path)
    end
    opts = options.data;
    opts.save_path = save_path;
    gen_data(opts)
    for la_it_method = 1:length(options.la.methods)
        la_method = options.la.methods{la_it_method};
        for la_power_order = reshape(options.la.power_orders,1,[])
            for la_rep_it = 1:options.la_repetition
                la_save_path = [save_path '/' la_method ...
                    '_' num2str(la_power_order,'%02d') ...
                    '_' num2str(la_rep_it,'%02d')]; 
                if not(exist(la_save_path,'dir'))
                    mkdir(la_save_path)
                end
                opts = options.la;
                opts.method = la_method;
                opts.data_path = save_path;
                opts.num_columns = options.data.num_columns;
                opts.power_order = la_power_order;
                opts.save_path = la_save_path;
                spectralla(opts);
            end
        end
    end
end
