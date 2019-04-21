function fig = structureplot_daily(pred_log_rho, nx, nz, xz)
% 
%

x = linspace(0, nx, nx);
z = linspace(0, nz, nz);
[X, Z] = meshgrid(x, z);
% levels = linspace(1, 3, 17);
levels = 17;


fig = figure;
contourf(X, Z, pred_log_rho, levels, 'LineStyle','none');
plot_electrode(xz)
title('Predictive resistivity');
xlabel('Width (m)'); ylabel('Depth (m)');
cbar = colorbar;
adjust_cbar(cbar);
ax = fig.CurrentAxes;
adjust_axis(ax);

aspect = [5, 1]; % figure aspect: [width, hight]
ratio = 240;
fig.Position = [100 100 ratio*aspect];

end

function plot_electrode(xz)
hold on;
plot(xz(:, 1), xz(:, 2), 'k.', 'MarkerSize', 12);
hold off;
end

function adjust_axis(ax)

ax.FontSize = 18; % All fontsize
ax.LineWidth = 1.2; % box width
ax.YDir = 'reverse';
ax.TickLength = [0.005, 0.025];
ax.Tag = 'ax';

end

function adjust_cbar(cbar)

colormap('jet');
caxis([1, 3]) % colorbar range
cbar.Label.String = '\Omega-m';
cbar.Ticks = [1, 1.5, 2, 2.5, 3];
cbar.TickLabels = {'10', '30', '100', '300', '1000'};
cbar.LineWidth = 0.8; % box and tick width
cbar.Tag = 'cbar';

end