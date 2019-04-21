function fig = heatmap_synth(synth_log_rho, pred_log_rho)
% plot heatmap of synthetic resistivity vs predictive resistivity
%
% Parameters
% ----------
% synth_log_rho : double
%     
% pred_log_rho : double
%    
% 
% Returns
% -------
% fig : figure graphics object
%     

%% setting
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

%% plotting
% set default fontsize
set_fontsize(DefaultTextFontSize, DefaultAxesFontSize);

% create figure
fig = figure('Position', fig_position);
ax = axes;
hold on; box on;

% heatmap
h = histogram2(synth_log_rho(:), pred_log_rho(:), ...
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