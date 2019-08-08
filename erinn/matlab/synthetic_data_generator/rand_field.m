function Targets = rand_field(para)
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
samples = gen_para.samples;    % number of samples <= 8000
mu_lower_bound = gen_para.mu_lower_bound;
mu_upper_bound = gen_para.mu_upper_bound;
std_lower_bound = gen_para.std_lower_bound;
std_upper_bound = gen_para.std_upper_bound;
x_kernel_size = gen_para.x_kernel_size;
z_kernel_size = gen_para.z_kernel_size;
sigma_size = [para.nx, para.nz]; % conductivity size

kernel_size = [x_kernel_size, z_kernel_size];
Targets = zeros(samples, para.nx * para.nz);
for i = 1:samples
    % The mean of the normal distribution is sampled 
    % from the uniform distribution.
    mu_ = (mu_upper_bound - mu_lower_bound) .* rand + mu_lower_bound;
    % The standard deviation of the normal distribution is sampled 
    % from the uniform distribution.
    std_ = (std_upper_bound - std_lower_bound) .* rand + std_lower_bound;
    % make a parent normal distribution for truncating.
    pd = makedist('Normal', 'mu', mu_, 'sigma', std_);
    % Truncate the distribution by restricting it to positive values.
    % Set the lower limit to 0 and the upper limit to infinity.
    trun_pd = truncate(pd, 0, inf);
    resistivity = random(trun_pd, sigma_size);
    % smoothing by moving average
    resistivity = conv2(resistivity, ones(kernel_size), 'same');
    normalize_matrix = conv2(ones(sigma_size), ones(kernel_size), 'same');
    resistivity = resistivity ./ normalize_matrix;
    sigma = 1 ./ resistivity;
    Targets(i, :) = sigma(:)';
end
