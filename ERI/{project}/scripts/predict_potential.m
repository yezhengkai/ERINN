clc; clear; close all;

%% add search path
toolbox_dir = fullfile('..', '..', '..', 'erinn');
addpath(genpath(toolbox_dir));

%% setting variables
% if you have daily data (urf), you can uncomment lines 12, 25~35.
prediction_dir = fullfile('..', 'models', 'predictions');
config_json = fullfile('..', 'config', 'config.json');
testing_h5 = fullfile(prediction_dir, 'testing.h5');
% daily_h5 = fullfile(prediction_dir, 'daily.h5');
para = jsondecode(fileread(config_json));

%% synthetic data: predict V/I
% synth_data is a structure data type
synth_data = load_synth_data(testing_h5);
pred_log_rho = synth_data.pred_log_rho;
sigma = 1./10.^pred_log_rho;
simu_para = fwd_simu_para(para);
pred_V = fwd_simu(sigma, simu_para);
save_synth_predV(testing_h5, pred_V);

%% daily data: predict V/I
% % daily_data is a table
% daily_data = load_daily_data(daily_h5);
% pred_log_rho = daily_data{:, 'pred_log_rho'};
% % or use: pred_log_rho = daily_data.pred_log_rho;
% sigma = 1./10.^pred_log_rho;
% simu_para = fwd_simu_para(para);
% pred_V = fwd_simu(sigma, simu_para);
% % *Because we have to use date information, we pass all table
% daily_data = addvars(daily_data, pred_V, 'Before','pred_log_rho');
% % or use: daily_data.pred_V = pred_V;
% save_daily_predV(daily_h5, daily_data);

%% remove search path
rmpath(genpath(toolbox_dir));