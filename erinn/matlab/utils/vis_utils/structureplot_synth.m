function fig = structureplot_synth(synth_log_rho, pred_log_rho,...
                             nx, nz, xz, varargin)
% Plot synthetic resistivity and predictive resistivity to illustrate 
% subsurface structure.
% 
% 
% Parameters
% ----------
% synth_log_rho : double, row vector or column vector
%     Synthetic resistivity.
% pred_log_rho : double, row vector or column vector
%     Predictive resisticity.
% nx : double
%     Number of mesh in x direction in forward model.
% nz : double
%     Number of mesh in z direction in forward model.
% xz : double
%     Electrode position in forward model.
% varargin : cell
%     Keyword arguments.
% 
% Returns
% -------
% fig : figure graphics object
%     We can post-process this figure before saving.
% 

%% parse keyword arguments
% default parameters for keyword arguments
default_metrics = {'None'};
default_scale_of_metrics = 'log';

% parse input arguments
p = inputParser;
metrics_set = {'mse', 'psnr', 'ssim'};
valid_metric = @(x) iscell(x) && ...
                    all(cellfun(@(y) (isstring(y) || ischar(y)), x)) && ...
                    all(cellfun(@(y) ismember(lower(y), metrics_set), x));
scale_set = {'log', 'linear'};
valid_scale = @(x) ischar(x) && ismember(x, scale_set);
addParameter(p, 'metrics', default_metrics, valid_metric);
addParameter(p, 'scale_of_metrics', default_scale_of_metrics, valid_scale);
parse(p, varargin{:});

%% set variables
metrics = p.Results.metrics;
scale_of_metrics = p.Results.scale_of_metrics;
x = [0, linspace(0.5, nx-0.5, nx), nx];
z = [0, linspace(0.5, nz-0.5, nz), nz];
[X, Z] = meshgrid(x, z);
% levels = linspace(1, 3, 17);
levels = 17;

%% Metrics
% scale of metrics
if strcmp(scale_of_metrics, 'linear')
    rho = 10 .^ pred_log_rho;
    rho_true = 10 .^ synth_log_rho;
else
    rho = pred_log_rho;
    rho_true = synth_log_rho;
end
% evaluate metrics
for i = 1:numel(metrics)
    if strcmpi(metrics{i}, 'MSE')
        metrics{i} = sprintf('MSE=%.4f', immse(rho, rho_true));
    elseif strcmpi(metrics{i}, 'PSNR')
        metrics{i} = sprintf('PSNR=%.4f', psnr(rho, rho_true));
    elseif strcmpi(metrics{i}, 'SSIM')
        metrics{i} = sprintf('SSIM=%.4f', ssim(rho, rho_true));
    end
end
metrics = strjoin(metrics, ', ');
if ~strcmp(metrics, 'None')
    title_msg = ['Predictive resistivity (', metrics, ')'];
else
    title_msg = 'Predictive resistivity';
end

%% plotting
fig = figure;
% synthetic
ax1 = subplot(2, 1, 1);
imagesc(0.5, 0.5, synth_log_rho);
plot_electrode(xz);
title('Synthetic resistivity');
ylabel('Depth (m)');
cbar1 = colorbar('Tag', 'cbar1');
adjust_cbar(ax1, cbar1);
ax1.Tag = 'ax1';
adjust_axis(ax1);


% extrapolation for plotting pretty figure
rho = zeros(nz + 2, nx + 2);
rho(2:end - 1, 2:end - 1) = pred_log_rho;
rho(1, 2:end - 1) = rho(2, 2:end - 1);
rho(end, 2:end-1) = rho(end - 1, 2:end-1);
rho(:, 1) = rho(:, 2);
rho(:, end) = rho(:, end - 1);
% predict
ax2 = subplot(2, 1, 2);
contourf(X, Z, rho, levels, 'LineStyle','none');
plot_electrode(xz)
title(sprintf(['Predictive resistivity -- ',...
               'PSNR=%.2f, ', 'SSIM=%.2f'], ...
               psnr(pred_log_rho, synth_log_rho), ...
               ssim(pred_log_rho, synth_log_rho)));
title(title_msg);
xlabel('Width (m)'); ylabel('Depth (m)');
cbar2 = colorbar('Tag', 'cbar2');
adjust_cbar(ax2, cbar2);
ax2.Tag = 'ax2';
ax2.YDir = 'reverse';
adjust_axis(ax2);

aspect = [5, 3]; % figure aspect: [width, hight]
ratio = 240;
fig.Position = [0 0 ratio*aspect];

end

function plot_electrode(xz)
hold on;
plot(xz(:, 1), xz(:, 2), 'k.', 'MarkerSize', 12);
hold off;
end

function adjust_axis(ax)
ax.FontSize = 18; % All fontsize
ax.LineWidth = 1.2; % box width
ax.TickLength = [0.005, 0.025];
end

function adjust_cbar(ax, cbar)
colormap(ax, 'jet');
caxis([1, 3]) % colorbar range
cbar.Label.String = '\Omega-m';
cbar.Ticks = [1, 1.5, 2, 2.5, 3];
cbar.TickLabels = {'10', '30', '100', '300', '1000'};
cbar.LineWidth = 0.8; % box and tick width
end
