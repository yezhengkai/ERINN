function Inputs = fwd_simu(Targets, simu_para)
% Simulate
%

Para = simu_para.Para;
dx = simu_para.dx;
dz = simu_para.dz;
recloc = simu_para.recloc;

% srcloc = simu_para.srcloc;
% dx = simu_para.dx;
% dz = simu_para.dz;
% recloc = simu_para.recloc;
% srcnum = simu_para.srcnum;
% Q = simu_para.Q;

% Inputs: delta V
samples = size(Targets, 1);
Inputs  = zeros(samples, size(recloc, 1));
sigma_size = [numel(dx), numel(dz)];

parfor ii = 1:samples
    s = reshape(Targets(ii, :), sigma_size);
%     [Para] = get_2_5Dpara_noQ(srcloc, dx, dz, s, 4, recloc, srcnum);
%     Para.Q = Q;
    [dobs, ~] = dcfw2_5D(s, Para);
    Inputs(ii,:) = dobs';
end