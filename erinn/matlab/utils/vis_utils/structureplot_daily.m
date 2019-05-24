function fig = structureplot_daily(pred_log_rho, nx, nz, xz)
% Plot predictive resistivity to illustrate subsurface structure.
% 
% 
% Parameters
% ----------
% pred_log_rho : double, row vector or column vector
%     Predictive resisticity.
% nx : double
%     Number of mesh in x direction in forward model.
% nz : double
%     Number of mesh in z direction in forward model.
% xz : double
%     Electrode position in forward model.
% 
% Returns
% -------
% fig : figure graphics object
%     We can post-process this figure before saving.
% 

x = [0, linspace(0.5, nx-0.5, nx), nx];
z = [0, linspace(0.5, nz-0.5, nz), nz];
[X, Z] = meshgrid(x, z);
% levels = linspace(1, 3, 17);
levels = 17;

% extrapolation for plotting pretty figure
rho = zeros(nz + 2, nx + 2);
rho(2:end - 1, 2:end - 1) = pred_log_rho;
rho(1, 2:end - 1) = rho(2, 2:end - 1);
rho(end, 2:end-1) = rho(end - 1, 2:end-1);
rho(:, 1) = rho(:, 2);
rho(:, end) = rho(:, end - 1);

fig = figure;
contourf(X, Z, rho, levels, 'LineStyle','none');
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