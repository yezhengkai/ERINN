function fig = heatmap_synth(synth_log_rho, pred_log_rho, varargin)
% Plot heatmap of synthetic resistivity vs predictive resistivity
%
% 
% Parameters
% ----------
% synth_log_rho : double
%     Synthetic resistivity.
% pred_log_rho : double
%     Predictive resisticity.
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
default_metrics = {'r^2', 'rmse', 'rmsre', 'corrcoef'};
default_scale_of_metrics = 'log';

% parse input arguments
p = inputParser;
metrics_set = {'r^2', 'mse', 'rmse', 'rmsre', 'corrcoef'};
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
DefaultTextFontSize = 22;
DefaultAxesFontSize = 22;
plt_linecolor = 'black';
plt_linewidth = 3;
plt_linestyle = '--';
cbar_linewidth = 0.8;
ax_colorscale = 'log';
ax_colormap = 'jet';
ax_linewidth = 1.5;

aspect = [1, 1]; % figure aspect: [width, hight]
ratio = 900;
fig_position = [150 50 ratio*aspect];
% fig_position = [150, 50, 900, 900];

%% Metrics
% scale of metrics
if strcmp(scale_of_metrics, 'linear')
    pred_log_rho = pred_log_rho(:);
    synth_log_rho = synth_log_rho(:);
    rho = 10 .^ pred_log_rho;
    rho_true = 10 .^ synth_log_rho;
else
    pred_log_rho = pred_log_rho(:);
    synth_log_rho = synth_log_rho(:);
    rho = pred_log_rho;
    rho_true = synth_log_rho;
end
% evaluate metrics
for i = 1:numel(metrics)
    if strcmpi(metrics{i}, 'MSE')
        MSE = immse(rho, rho_true);
        metrics{i} = sprintf('MSE: %6.4f', MSE);
    elseif strcmpi(metrics{i}, 'RMSE')
        RMSE = sqrt(mean((rho_true - rho) .^ 2));
        metrics{i} = sprintf('RMSE: %6.4f', RMSE);
    elseif strcmpi(metrics{i}, 'R^2')
        Ybar = mean(rho_true);
        SS_tot = sum((rho_true - Ybar) .^ 2);
        SS_res = sum((rho_true - rho) .^ 2);
        R2 = 1 - SS_res/SS_tot;
        metrics{i} = sprintf('R^2: %6.4f', R2);
    elseif strcmpi(metrics{i}, 'RMSRE')
        RMSRE = sqrt(mean(((rho_true - rho) ./ rho_true) .^ 2));
        metrics{i} = ...
            sprintf('RMSRE: %6.4f', RMSRE);
    elseif strcmpi(metrics{i}, 'corrcoef')
        C = corrcoef(rho_true, rho);
        metrics{i} = sprintf('corrcoef: %6.4f', C(1, 2));
    end
end

%% plotting
% set default fontsize
set_fontsize(DefaultTextFontSize, DefaultAxesFontSize);

% create figure
fig = figure('Position', fig_position);
ax = axes;
hold on; box on;

% heatmap
h = histogram2(synth_log_rho, pred_log_rho, ...
    'DisplayStyle', 'tile');
h.BinWidth = [0.05, 0.05];

% diagonal line
% lower_bound = min([h.XBinLimits, h.YBinLimits]);
% upper_bound = max([h.XBinLimits, h.YBinLimits]);
x_min = h.XBinLimits(1);
x_max = h.XBinLimits(2);
R = 0.1;
lower_bound = x_min - (x_max - x_min) * R;
upper_bound = x_max + (x_max - x_min) * R;
while lower_bound > min([h.XBinLimits, h.YBinLimits]) || ...
        upper_bound < max([h.XBinLimits, h.YBinLimits])
    R = R + 0.05;
    lower_bound = x_min - (x_max - x_min) * R;
    upper_bound = x_max + (x_max - x_min) * R;
end
plot(ax, [lower_bound, upper_bound], [lower_bound, upper_bound], ...
    'LineWidth', plt_linewidth, ...
    'Color', plt_linecolor, ...
    'LineStyle', plt_linestyle, ...
    'Tag', 'ax');

% adjust colorbar
cbar = colorbar('Tag', 'cbar');
colormap(ax, ax_colormap);
cbar.LineWidth = cbar_linewidth; % box and tick width
ax.ColorScale = ax_colorscale;

% adjust axes
ax.LineWidth = ax_linewidth; % box and tick width
ax.XLim = [lower_bound, upper_bound];
ax.YLim = [lower_bound, upper_bound];

% label string
xlabel('Synthetic resistivity log_{10}(\Omega-m)');
ylabel('Predictive resistivity log_{10}(\Omega-m)');
cbar.Label.String = 'Count';

% metrics text string
metrics_str = sprintf(strjoin(metrics, '\n'));
text(lower_bound + 0.05 * (upper_bound - lower_bound), ...
     upper_bound - 0.05 * (upper_bound - lower_bound), ...
     metrics_str,...
     'FontName', 'Bookman',...
     'HorizontalAlignment', 'left',...
     'VerticalAlignment', 'top',...
     'BackgroundColor', [1, 0.5, 0.5],...
     'Margin', 5);

% remove default fontsize
rm_fontsize()
% cbar.FontSize = fontsize;
% cbar.Label.FontSize = fontsize * ax.LabelFontSizeMultiplier;
% ax.FontSize = fontsize;
end


function set_fontsize(DefaultTextFontSize, DefaultAxesFontSize)
set(groot, 'DefaultTextFontSize', DefaultTextFontSize);
set(groot, 'DefaultAxesFontSize', DefaultAxesFontSize);
end


function rm_fontsize()
set(groot, 'DefaultTextFontSize', 'remove');
set(groot, 'DefaultAxesFontSize', 'remove');
end