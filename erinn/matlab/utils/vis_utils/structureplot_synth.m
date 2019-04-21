function fig = structureplot_synth(synth_log_rho, pred_log_rho,...
                             nx, nz, xz)
% TODO: 根據nx, nz來調整比例 => 轉換公式
% config = jsondecode(fileread(config_json));
% nx = config.nx;
% nz = config.nz;
% fig = structureplot_synth(synth_log_rho, pred_log_rho, nx, nz, xz)
% fig.Position = [left, bottom, width, height]

x = linspace(0, nx, nx);
z = linspace(0, nz, nz);
[X, Z] = meshgrid(x, z);
% levels = linspace(1, 3, 17);
levels = 17;


fig = figure;

ax1 = subplot(2, 1, 1);
imagesc(0.5, 0.5, synth_log_rho);
plot_electrode(xz);
title('Synthetic resistivity');
ylabel('Depth (m)');
cbar1 = colorbar('Tag', 'cbar1');
adjust_cbar(ax1, cbar1);
ax1.Tag = 'ax1';
adjust_axis(ax1);


ax2 = subplot(2, 1, 2);
contourf(X, Z, pred_log_rho, levels, 'LineStyle','none');
plot_electrode(xz)
title('Predictive resistivity');
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