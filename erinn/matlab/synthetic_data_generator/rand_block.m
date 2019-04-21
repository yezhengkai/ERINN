function Targets = rand_block(para)
% Generate random block.
%
% Parameters
% ----------
% para : struct
% 
% 
% Returns
% -------
% Targets : double
%     2D matrix which rows are samples and columns are features.

gen_para = para.generator;

% Target: true resistivity
samples = gen_para.samples;    % samples <= 8000
sigma_size = [para.nx, para.nz]; % conductivity size
Targets = zeros(samples, para.nx * para.nz);
X = para.nx;  % model total x width, because resolution is 1m*1m
Z = para.nz;  % model total z depth, because resolution is 1m*1m
x_width_range = [gen_para.block_x_min, gen_para.block_x_max];
z_width_range = [gen_para.block_z_min, gen_para.block_z_max];


if any(cellfun(@(x) strcmp(x, 'mu'), fieldnames(gen_para)))
    mu = gen_para.mu;       % Mean of log(X)
    std = gen_para.std;      % St. dev. of log(X)
    Sigma = 1./lognrnd(mu, std, samples, 1);
    sigma_abnormal = 1./lognrnd(mu, std, samples, 1);
elseif any(cellfun(@(x) strcmp(x, 'lower_bound'), fieldnames(gen_para)))
    lb = gen_para.lower_bound;    % rho of log10 scale
    ub = gen_para.upper_bound;    % rho of log10 scale
    Sigma = 1 ./ (10 .^ ((ub - lb) .* rand(samples, 1) + lb));
    sigma_abnormal = 1 ./ (10 .^ ((ub - lb) .* rand(samples, 1) + lb));
end


% Sigma = 1./lognrnd(mu, std, samples, 1);
% sigma_abnormal = 1./lognrnd(mu, std, samples, 1);

% TODO: Use parfor. if we use pardor, we should set rand stream.
% generate block structure
for i = 1:samples
    x_width = randi(x_width_range, 1);
    z_width = randi(z_width_range, 1);
    x = randi([1, X-x_width], 1);
    z = randi([1, Z-z_width], 1);
%     mu = -2;     % Mean of log10(X)
%     std = 0.5;      % St. dev. of log10(X)
%     sigma = power(10, mu + std*randn(1));
%     sigma_abnormal = power(10, mu + std*randn(1));
    s = Sigma(i) * ones(sigma_size);
    s(x+1:x+x_width, z+1:z+z_width) = sigma_abnormal(i);
    Targets(i,:) = s(:)';
end

%% doc lognrnd
% The normal and lognormal distributions are closely related. 
% If X is distributed lognormally with parameters ? and £m, 
% then log(X) is distributed normally with mean ? and standard deviation £m.

% The mean m and variance v of a lognormal random variable are 
% functions of ? and £m that can be calculated with the lognstat function.
% They are:

% m = 1; % m = exp(mu + sigma^2 / 2);
% v = 2; % v = exp(2 * mu + sigma^2) * (exp(sigma^2) - 1);
% mu = log((m^2) / sqrt(v + m^2));
% sigma = sqrt(log(v / (m^2) + 1));
% 
% [M, V] = lognstat(mu, sigma)
% 
% X = lognrnd(mu, sigma, 1, 1e6);
% 
% MX = mean(X)
% 
% VX = var(X)
% 
% histogram(X);
% histogram(log(X));
