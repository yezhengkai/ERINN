function [srcloc, dx, dz, recloc, srcnum] = prepare_for_get_2_5Dpara(para)
%%Detailed comment is in get_2_5Dpara!
% recloc=(P1x, P1z, P2x, P2z). z is positive down.

simu_para = para.simulate;
%% parameters for FW2_5D
[Tx_id, Rx_id, ~, coord, data] = read_urf(simu_para.geometry_urf);

% Collect pairs id
if isempty(data)
    C_pair = nchoosek(Tx_id, 2);
    P_pair = nchoosek(Rx_id, 2);
    
    % https://stackoverflow.com/questions/21312231/efficiently-converting-java-list-to-matlab-matrix
    % use java linking list
    % Generic fast method for creating CP_pair
    CP_pair = java.util.LinkedList;
    for i = 1:size(C_pair, 1)
        for j = 1:size(P_pair, 1)
            if ~ismember(C_pair(i, :), P_pair(j, :))
                CP_pair.add([C_pair(i, :), P_pair(j, :)]);
            end
        end
    end
    CP_pair = cell2mat(cellfun(@(x) transpose(x), ...
                               cell(CP_pair.toArray), ...
                               'UniformOutput', false));
else
    CP_pair = data(:, 1:4);
end

% Convert id to coordinate
recloc = horzcat(coord(CP_pair(:, 3), [2, 4]), coord(CP_pair(:, 4), [2, 4]));
recloc(:, [2, 4]) = abs(recloc(:, [2, 4])); % z is positive down.
SRCLOC = horzcat(coord(CP_pair(:, 1), [2, 4]), coord(CP_pair(:, 2), [2, 4]));
SRCLOC(:, [2, 4]) = abs(SRCLOC(:, [2, 4])); % z is positive down.

% Collect pairs that fit the array configuration
if ~strcmp(simu_para.array_type, 'CPP')
    % Check if the electrode is on the ground
    at_ground = and(and(SRCLOC(:, 2) == 0, SRCLOC(:, 4) == 0), ...
                and(recloc(:, 2) == 0, recloc(:, 4) == 0));
    SRCLOC = SRCLOC(at_ground, :);
    recloc = recloc(at_ground, :);
    AM = recloc(:, 1) - SRCLOC(:, 1);
    MN = recloc(:, 3) - recloc(:, 1);
    NB = SRCLOC(:, 3) - recloc(:, 3);
    % Check that the electrode arrangement is correct
    positive_idx = and(and(AM > 0, MN > 0), NB > 0);
    SRCLOC = SRCLOC(positive_idx, :);
    recloc = recloc(positive_idx, :);
    AM = AM(positive_idx);
    MN = MN(positive_idx);
    NB = NB(positive_idx);
    switch simu_para.array_type
        case 'Wenner_Schlumberger'
            % Must be an integer multiple?
            row_idx = and(AM == NB, ...
                          arrayfun(@(x, y) mod(x, y), AM, MN) == 0);
            SRCLOC = SRCLOC(row_idx, :);
            recloc = recloc(row_idx, :);
        case 'Wenner' 
            row_idx = and(AM == MN, MN == NB);
            SRCLOC = SRCLOC(row_idx, :);
            recloc = recloc(row_idx, :);
        case 'Wenner_Schlumberger_NonInt'
            row_idx = and(AM == NB, AM >= MN);
            SRCLOC = SRCLOC(row_idx, :);
            recloc = recloc(row_idx, :);
        %case 'Schlumberger'
        %    % Must be an integer multiple?
        %    row_idx = and(and(AM == NB, AM > MN), ...
        %                  arrayfun(@(x, y) mod(x, y), AM, MN) == 0);
        %    SRCLOC = SRCLOC(row_idx, :);
        %    recloc = recloc(row_idx, :);
    end
end
[srcloc, ~, srcnum] = unique(SRCLOC, 'rows', 'stable');

array_len = max(coord(:, 2)) - min(coord(:, 2));
srcloc(:, [1, 3]) = srcloc(:, [1, 3]) - array_len/2;
recloc(:, [1, 3]) = recloc(:, [1, 3]) - array_len/2;
dx = ones(para.nx, 1);
dz = ones(para.nz, 1);

%% testing
% https://www.mathworks.com/matlabcentral/answers/21004-is-there-an-elegant-way-to-create-dynamic-array-in-matlab
% use java stack
%tic;
%tmp = java.util.Stack();
%for i = 1:size(C_pair, 1)
%   for j = 1:size(P_pair, 1)
%       if ~ismember(C_pair(i, :), P_pair(j, :))
%           tmp.push([C_pair(i, :), P_pair(j, :)]);
%       end
%   end
%end
%A = cell2mat(cellfun(@(x) transpose(x), cell(tmp.toArray), ...
%   'UniformOutput', false));
%toc;
% Elapsed time is 19.548198 seconds.
% Elapsed time is 18.822577 seconds.

% https://stackoverflow.com/questions/21312231/efficiently-converting-java-list-to-matlab-matrix
% use java linking list
%tic;
%tmp = java.util.LinkedList;
%for i = 1:size(C_pair, 1)
%   for j = 1:size(P_pair, 1)
%       if ~ismember(C_pair(i, :), P_pair(j, :))
%           tmp.add([C_pair(i, :), P_pair(j, :)]);
%       end
%   end
%end
%B = cell2mat(cellfun(@(x) transpose(x), cell(tmp.toArray), ...
%   'UniformOutput', false));
%toc;
% Elapsed time is 18.040336 seconds.
% Elapsed time is 16.393513 seconds.

% Trash
%tic;
%tmp = [];
%for i = 1:size(C_pair, 1)
%   for j = 1:size(P_pair, 1)
%       if ~ismember(C_pair(i, :), P_pair(j, :))
%           tmp = [tmp; [C_pair(i, :), P_pair(j, :)]];
%       end
%   end
%end
%toc;
% Elapsed time is 1166.231252 seconds.

% The following method is specific to the case 
% when Tx_id and Rx_id do not intersection.
% C = cell2mat(arrayfun(@(a, r) repmat(a, r, 1),...
%                       C_pair, ones(size(C_pair)) * length(P_pair), ...
%                       'UniformOutput', false));
% P = repmat(P_pair, length(C_pair), 1);

% cell2mat(arrayfun(@(a, r) repmat(a, r, 1),...
%                   C_pair, ones(size(C_pair)) * length(P_pair),...
%                   'UniformOutput', false))
% equivalent to repelem(tx_pair, length(rx_pair), 1); (Introduced in R2015a)
% CP_pair = [cell2mat(arrayfun(@(a, r) repmat(a, r, 1),...
%                     C_pair, ones(size(C_pair)) * length(P_pair),...
%                     'UniformOutput', false)), ...
%                     repmat(P_pair, length(C_pair), 1)];
