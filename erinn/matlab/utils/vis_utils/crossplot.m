function fig = crossplot(true_value, pred_value, varargin)
% Crossplot of synthetic V/I versus predictive V/I.
% 
% 
% Parameters
% ----------
% true_value : double, row vector or column vector
%     True(obsverved or synthetic) value. 
% pred_value : double, row vector or column vector
%     Predictive value.
% varargin : cell
%     Keyword arguments.
% 
% Returns
% -------
% fig : figure graphics object
%     We can post-process this figure before saving.
%
% References
% ----------
% https://en.wikipedia.org/wiki/Coefficient_of_determination
% https://www.mathworks.com/matlabcentral/answers/378389-calculate-r-squared-from-a-linear-regress
% https://www.mathworks.com/matlabcentral/fileexchange/15130-error-related-performance-metrics
% https://stats.stackexchange.com/questions/260615/what-is-the-difference-between-rrmse-and-rmsre

%% parse keyword arguments
% default parameters for keyword arguments
default_metrics = {'r^2', 'rmse', 'rmsre', 'corrcoef', 'nan_ratio'};
default_type = {'scatter'};
default_xlabel = 'True value';
default_ylabel = 'Predictive value';

% parse input arguments
p = inputParser;
metrics_set = {'r^2', 'rmse', 'rmsre', 'corrcoef', 'nan_ratio'};
valid_metric = @(x) iscell(x) && ...
                    all(cellfun(@(y) (isstring(y) || ischar(y)), x)) && ...
                    all(cellfun(@(y) ismember(lower(y), metrics_set), x));
type_set = {'scatter', 'histogram2'};
valid_type = @(x) ischar(x) && ismember(x, type_set);
valid_xlabel = @(x) ischar(x);
valid_ylabel = @(x) ischar(x);
addParameter(p, 'metrics', default_metrics, valid_metric);
addParameter(p, 'plot_type', default_type, valid_type);
addParameter(p, 'xlabel', default_xlabel, valid_xlabel);
addParameter(p, 'ylabel', default_ylabel, valid_ylabel);
parse(p, varargin{:});


%% set variables
% for keyword arguments
metrics = p.Results.metrics;
plot_type = p.Results.plot_type;
xlabel_str = p.Results.xlabel;
ylabel_str = p.Results.ylabel;
% for plotting
DefaultTextFontSize = 22;
DefaultAxesFontSize = 22;
extended_ratio = 0.1;
plt_linecolor = 'red';
plt_linewidth = 3;
plt_linestyle = '-';
cbar_linewidth = 0.8;
ax_colorscale = 'log';
ax_colormap = 'jet';
ax_linewidth = 1.5;
ax_TickDir = 'out';

aspect = [1, 1]; % figure aspect: [width, hight]
ratio = 900;
fig_position = [150 50 ratio*aspect];


%% Metrics
% evaluate metrics
for i = 1:numel(metrics)
    if strcmpi(metrics{i}, 'R^2')
        Ybar = mean(true_value);
        SS_tot = nansum((true_value - Ybar) .^ 2);
        SS_res = nansum((true_value - pred_value) .^ 2);
        R2 = 1 - SS_res/SS_tot;
        metrics{i} = sprintf('R^2: %6.4f', R2);
    elseif strcmpi(metrics{i}, 'RMSE')
        RMSE = sqrt(nanmean((true_value - pred_value) .^ 2));
        metrics{i} = sprintf('RMSE: %6.4f', RMSE);
    elseif strcmpi(metrics{i}, 'RMSRE')
        RMSRE = sqrt(nanmean(((true_value - pred_value) ./ true_value) .^ 2));
        metrics{i} = sprintf('RMSRE: %6.4f', RMSRE);
    elseif strcmpi(metrics{i}, 'corrcoef')
        C = corrcoef(true_value(~isnan(true_value)), ...
                     pred_value(~isnan(pred_value)));
        metrics{i} = sprintf('corrcoef: %6.4f', C(1, 2));
    elseif strcmpi(metrics{i}, 'nan_ratio')
        num_nan = sum(isnan(true_value));
        num_obs_V = length(true_value);
        nan_ratio = 100 * num_nan/num_obs_V;
        if nan_ratio == 100
            warning('nan ratio is 1. The crossplot was not drawn.');
            return
        end
        metrics{i} = sprintf('nan ratio: %.2f%%%', nan_ratio);
    end
end


%% crossplot
% set default fontsize
set_fontsize(DefaultTextFontSize, DefaultAxesFontSize);

% get maximum and minimum of x direction
x_min = nanmin(true_value);
x_max = nanmax(true_value);
lower_bound = x_min - (x_max - x_min) * extended_ratio;
upper_bound = x_max + (x_max - x_min) * extended_ratio;

% basemap plot (scatter or histogram2)
fig = figure('Position', fig_position);
ax = axes;
hold on; box on;
if strcmpi(plot_type, 'scatter')
    scatter(true_value, pred_value, 'filled',...
        'MarkerEdgeAlpha', 0,...
        'MarkerFaceAlpha', 0.6);
elseif strcmpi(plot_type, 'histogram2')
    histogram2(true_value, pred_value, ...
              'BinWidth', ...
              [(x_max - x_min) / 150, (x_max - x_min) / 150], ...
              'DisplayStyle', 'tile', ...
              'EdgeAlpha', 0);
end

% diagonal line
plot(ax, [x_min, x_max], [x_min, x_max], ...
     'LineWidth', plt_linewidth, ...
     'Color', plt_linecolor, ...
     'LineStyle', plt_linestyle, ...
     'Tag', 'ax');

% adjust colorbar
if strcmpi(plot_type, 'histogram2')
    cbar = colorbar('Tag', 'cbar');
    old_caxis = caxis;
    caxis(ax, [1 old_caxis(2)]); 
    colormap(ax, ax_colormap);
    cbar.LineWidth = cbar_linewidth; % box and tick width
    ax.ColorScale = ax_colorscale;
end

% adjust axes
ax.FontSize = 22; % points
ax.TickDir = ax_TickDir;
ax.LineWidth = ax_linewidth;
ax.XLim = [lower_bound, upper_bound];
ax.YLim = [lower_bound, upper_bound];
% if length(ax.XTick) == length(ax.YTick)
%     ax.YTick = ax.XTick;
% elseif length(ax.XTick) < length(ax.YTick)
%     ax.YTick = ax.XTick;
% elseif length(ax.XTick) > length(ax.YTick)
%     ax.XTick = ax.YTick;
% end


% label string
xlabel(xlabel_str);
ylabel(ylabel_str);
cbar.Label.String = 'Count';

% use text and then use annotation plot background color
metrics_str = strjoin(metrics, '\n');
text(lower_bound + 0.05 * (upper_bound - lower_bound), ...
     upper_bound - 0.05 * (upper_bound - lower_bound), ...
     metrics_str,...
     'FontName', 'Bookman',...
     'FontSize', 22,...
     'HorizontalAlignment', 'left',...
     'VerticalAlignment', 'top',...
     'BackgroundColor', [1, 0.5, 0.5],...
     'Margin', 5);
 
% remove default fontsize
rm_fontsize()

end

function set_fontsize(DefaultTextFontSize, DefaultAxesFontSize)
set(groot, 'DefaultTextFontSize', DefaultTextFontSize);
set(groot, 'DefaultAxesFontSize', DefaultAxesFontSize);
end


function rm_fontsize()
set(groot, 'DefaultTextFontSize', 'remove');
set(groot, 'DefaultAxesFontSize', 'remove');
end