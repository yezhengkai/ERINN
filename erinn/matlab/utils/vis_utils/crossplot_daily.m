function fig = crossplot_daily(obs_V, pred_V, varargin)
% Crossplot of synthetic V/I versus predictive V/I.
% 
% 
% Parameters
% ----------
% synth_V : double, row vector or column vector
%     Synthetic delta V/I.
% pred_V : double, row vector or column vector
%     Predictive delta V/I.
%     We can get this parameter by forward simulation,
%     where the input is the resistivity predicted by NN.
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

%% calculate metrics
% 1. R squared
Ybar = mean(obs_V);
SS_tot = nansum((obs_V - Ybar).^2);
% SS_reg = nansum((pred_V - Ybar).^2);
SS_res = nansum((obs_V - pred_V).^2);
R2 = 1 - SS_res/SS_tot;
% 2. Root mean square
RMS = sqrt(nanmean((obs_V - pred_V).^2));
% 3. Root mean squared relative error
RMSRE = sqrt(nanmean(((obs_V - pred_V) ./ obs_V) .^ 2));
% 4. correlation coefficient
C = corrcoef(obs_V, pred_V);
% 5. nan ratio
num_nan = sum(isnan(obs_V));
num_obs_V = length(obs_V);
nan_ratio = num_nan/num_obs_V;
if nan_ratio == 1
    warning('nan ratio is 1. The crossplot is not drawn.');
    return
end

%% get obs_V maximum and minimum
x_min = nanmin(obs_V); x_max = nanmax(obs_V);
if abs(x_max) > abs(x_min)
    lim = abs(x_max);
else
    lim = abs(x_min);
end
r = 2 * lim;

%% crossplot
fig = figure('Position', [150, 50, 900, 900]);

% scatter plot
scatter(obs_V, pred_V, 'filled',...
    'MarkerEdgeAlpha', 0,...
    'MarkerFaceAlpha', 0.6);
% adjust axes
ax = fig.CurrentAxes;
ax.FontSize = 22; % points
ax.TickDir = 'out';
ax.Box = 'on';
ax.LineWidth = 1.5;

% diagonal line
hold on;
plt = plot([x_min, x_max], [x_min, x_max], 'r', 'Parent', ax);
plt.LineWidth = 3;
axis([-lim * 1.1, lim * 1.1, -lim * 1.1, lim * 1.1]);
xlabel('Observed \DeltaV/I');
ylabel('Predictive \DeltaV/I');

% use text and then use annotation plot background color
str = sprintf(['R^2: %6.4f\n',...
               'RMS: %6.4f\n',...
               'RMSRE: %6.4f\n',...
               'corrcoef: %6.4f\n',...
               'nan ratio: %.2f%%'],...
               R2, RMS, RMSRE, C(1, 2), nan_ratio);
text(-lim + 0.01 * r, lim - 0.01 * r, str,...
    'FontName', 'Bookman',...
    'FontSize', 22,...
    'HorizontalAlignment', 'left',...
    'VerticalAlignment', 'top',...
    'BackgroundColor', [1, 0.5, 0.5],...
    'Margin', 5);




%% some code snippet
% % calculate metrics
% Ybar = mean(synth_V(i, :));
% SS_tot = sum((synth_V(i, :) - Ybar).^2);
% % SS_reg = sum((pred_V(i, :) - Ybar).^2);
% SS_res = sum((synth_V(i, :) - pred_V(i, :)).^2);
% R2 = 1 - SS_res/SS_tot;
% RMS = sqrt(nanmean((synth_V(i, :) - pred_V(i, :)).^2));
% C = corrcoef(synth_V(i, :), pred_V(i, :));
% 
% % get synth_V maximum and minimum
% x_min = nanmin(synth_V(i, :)); x_max = nanmax(synth_V(i, :));
% if abs(x_max) > abs(x_min)
%     lim = abs(x_max);
% else
%     lim = abs(x_min);
% end
% r = 2 * lim;
% 
% % ---- plot from here ----
% fig = figure('Units', 'inches', 'Position', [1.5, 1.5, 8.25, 8.25]);
% s = scatter(synth_V(i, :), pred_V(i, :), 'filled',...
%     'MarkerEdgeAlpha', 0, 'MarkerFaceAlpha', 0.6);
% ax = subplot(1, 1, 1, 'FontSize', 22, 'TickDir', 'out',...
%     'Box', 'on', 'LineWidth', 1.5);
% %     ax = axes;
% 
% % scatter plot
% %     s = scatter(ax, synth_V(i, :), pred_V(i, :), 'filled',...
% %         'MarkerEdgeAlpha', 0, 'MarkerFaceAlpha', 0.6);
% %     s.MarkerEdgeAlpha = 0;
% %     s.MarkerFaceAlpha = 0.6;
% 
% % diagonal line
% hold on;
% plt = plot([x_min, x_max], [x_min, x_max], 'r', 'Parent', ax);
% plt.LineWidth = 3;
% axis([-lim * 1.1, lim * 1.1, -lim * 1.1, lim * 1.1]);
% xlabel('Synthetic \DeltaV/I');
% ylabel('Predictive \DeltaV/I');
% 
% % use text and then use annotation plot background color
% str = sprintf(' R^2: %6.4f\n RMS: %6.4f\n corrcoef: %6.4f',...
%     R2, RMS, C(1, 2));
% t = text(-lim + 0.01 * r, lim - 0.01 * r, str,...
%     'FontName', 'Latin Modern Math',...
%     'FontSize', 22, 'HorizontalAlignment', 'left',...
%     'VerticalAlignment', 'top',...
%     'BackgroundColor', [1, 0.5, 0.5],...
%     'Margin', 5);
%     t.FontName = 'Latin Modern Math';
%     t.FontSize = 22;
%     t.HorizontalAlignment = 'left';
%     t.VerticalAlignment = 'top';
%     t.BackgroundColor = [1, 0.5, 0.5];
%     t.Margin = 5;

% % annotation
% str = sprintf(' R^2: %6.4f\n RMS: %6.4f\n corrcoef: %6.4f',...
%     R2, RMS, C(1, 2));
% dim = ax.Position;
% anno = annotation(fig, 'textbox', dim, 'String', str,...
%     'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom');
    
% adjust annotation
% anno.FontName = 'Cambria Math';
% anno.FontName = 'Latin Modern Math';
% anno.FontSize = 22;
% anno.FitBoxToText = 'on';
% anno.EdgeColor = 'None';
% anno.BackgroundColor = [1, 0.5, 0.5];
% anno.FaceAlpha = 0.8;
% anno.Margin = 12;
% anno.Position = [ax.Position([1, 2]) ax.Position([1, 2]);
% anno.HorizontalAlignment = 'left';
% anno.VerticalAlignment = 'top';
%     
% % adjust axes
% ax.FontSize = 22; % points
% ax.TickDir = 'out';
% ax.Box = 'on';
% ax.LineWidth = 1.5;
% 
% % adjust fig
% fig.Position = [150, 100, 900, 900];