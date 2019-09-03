clc; clear; close all;

%% add search path
toolbox_dir = fullfile('..', '..', '..', 'erinn');
addpath(genpath(toolbox_dir));

%% setting
generate_config_json = true;
config_json = fullfile('..', 'config', 'config.json'); %創建資料夾的方法，可參考
urf = fullfile('..', 'config', 'geo.urf');
array_type = 'CPP';
core = 'rand_block';
pdf = 'log10_uniform'; %以10為底數的log均勻分布

h5 = fullfile('..', 'config', 'glob_para.h5');
num_h5 = 40; %跑40次

%% generate config json
% Use following comments to create a default config file.
% The parameters in the config file must be checked to suit your situation.
if generate_config_json
    
    gen_config_json('output_json', config_json, ...
        'urf', urf, ...
        'array_type', array_type, ...
        'core', core, ...
        'pdf', pdf);
    
    msg = strcat('==========================================\n', ...
                 'If the configuration has been confirmed,\n', ...
                 'enter Y to continue the program, \n', ...
                 'or enter another value to end the program.\n');
    resp = input(msg, 's');
    if ~strcmpi(resp, 'y')
        fprintf('Exit!\n')
        return
    end
end

%% generate required data
% generate synthetic data
for i = 1:num_h5
    gen_data(config_json); %總共會跑出samples*num_h5個數量的model
end

% generate global parameters
gen_glob_para_h5(h5, config_json);

%% remove search path
rmpath(genpath(toolbox_dir));
