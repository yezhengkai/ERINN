function simu_para = fwd_simu_para(para)

simu_para = para.simulate;
%% Prepare parameters for get_2_5Dpara
[srcloc, dx, dz, recloc, srcnum] = prepare_for_get_2_5Dpara(para);
% [Tx_id, Rx_id, ~, coord, data] = read_urf(simu_para.geometry_urf);
% if isempty(data)
%     C_pair = nchoosek(Tx_id, 2);
%     P_pair = nchoosek(Rx_id, 2);
% %     C = cell2mat(arrayfun(@(a, r) repmat(a, r, 1),...
% %             C_pair, ones(size(C_pair)) * length(P_pair), 'UniformOutput', false));
% %     P = repmat(P_pair, length(C_pair), 1);
%     % equivalent to repelem(tx_pair, length(rx_pair), 1); (Introduced in R2015a)
%     CP_pair = [cell2mat(arrayfun(@(a, r) repmat(a, r, 1),...
%                         C_pair, ones(size(C_pair)) * length(P_pair),...
%                         'UniformOutput', false)), ...
%                         repmat(P_pair, length(C_pair), 1)];
% else
%     CP_pair = data(:, 1:4);
% end
% 
% recloc = horzcat(coord(CP_pair(:, 3), [2, 4]), coord(CP_pair(:, 4), [2, 4]));
% recloc(:, [2, 4]) = abs(recloc(:, [2, 4]));
% srcloc = horzcat(coord(CP_pair(:, 1), [2, 4]), coord(CP_pair(:, 2), [2, 4]));
% [srcloc, ia, srcnum] = unique(srcloc, 'rows', 'stable');
% srcloc(:, [2, 4]) = abs(srcloc(:, [2, 4]));
% 
% array_len = max(coord(:, 2)) - min(coord(:, 2));
% srcloc(:, [1, 3]) = srcloc(:, [1, 3]) - array_len/2;
% recloc(:, [1, 3]) = recloc(:, [1, 3]) - array_len/2;
% dx = ones(para.nx, 1);
% dz = ones(para.nz, 1);

%% prepare Para structure, because it is time consuming.
if ~exist(simu_para.Para_mat, 'file')
    fprintf('Create Para.\n');
    s = ones(para.nx, para.nz);
    [Para] = get_2_5Dpara(srcloc, dx, dz, s, 4, recloc, srcnum);
    save(simu_para.Para_mat, 'Para');
    simu_para.Para = Para;
else
    fprintf('Load Para mat file.\n');
    load(simu_para.Para_mat);
    % Check if Para is in accordance with current configuration
    if size(Para.Q, 1) ~= size(srcnum, 1) || ...
            size(Para.Q, 2) ~= numel(dx) * numel(dz) || ...
            size(Para.b, 2) ~= size(srcloc, 1)
        fprintf('Size of Q matrix is wrong, creating a new one.\n');
        s = ones(para.nx, para.nz);
        [Para] = get_2_5Dpara(srcloc, dx, dz, s, 4, recloc, srcnum);
        save(simu_para.Para_mat, 'Para');
        simu_para.Para = Para;
    else
        simu_para.Para = Para;
    end
end

%%
simu_para.srcloc = srcloc;
simu_para.dx = dx;
simu_para.dz = dz;
simu_para.recloc = recloc;
simu_para.srcnum = srcnum;
