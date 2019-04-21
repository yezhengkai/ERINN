clc; clear; close all;

%% add search path
toolbox_dir = fullfile('..', '..', '..', 'erinn');
addpath(genpath(toolbox_dir));

%% setting variables
prediction_dir = fullfile('..', 'models', 'predictions');
config_json = fullfile('..', 'config', 'config.json');
testing_h5 = fullfile(prediction_dir, 'testing.h5');
para = jsondecode(fileread(config_json));

%% synthetic data: predict V/I
% synth_data is a structure data type
synth_data = load_synth_data(testing_h5);
pred_log_rho = synth_data.pred_log_rho;
sigma = 1./10.^pred_log_rho;
simu_para = fwd_simu_para(para);
pred_V = fwd_simu(sigma, simu_para);
save_synth_predV(testing_h5, pred_V);

%% remove search path
rmpath(genpath(toolbox_dir));