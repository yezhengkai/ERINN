function block_para = get_block_para(h5_list)
% Get the block parameters in synthetic resistivity model
% 
% Parameters
% ----------
% h5_list : cell of char or string array
% 
% Returns
% -------
% block_para : table
% 
% 

dx = h5read(h5_list{1}, '/dx');
nx = numel(dx);
dz = h5read(h5_list{1}, '/dz');
nz = numel(dz);
num_h5 = numel(h5_list);
block_para = cell(num_h5, 1);
block_heatmap = zeros(nz, nx);
for n = 1:num_h5
    
    rho = 1 ./ h5read(h5_list{n}, '/Targets');
    
    tmp_paras = cell(size(rho, 1), 1);  % in one hdf5 file
    for i = 1:size(rho, 1)
        tmp = reshape(rho(i, :), nx, nz)';
        rho_vals = unique(tmp);
        
        tmp_para = cell(numel(rho_vals), 1);  % in one synthetic model
        for j = 1:numel(rho_vals)
            [row, col] = find(tmp == rho_vals(j));
            min_row = min(row);
            max_row = max(row);
            min_col = min(col);
            max_col = max(col);
                        
            
            % method 1
            %tmp = false(numel(unique(row)), numel(unique(col)));
            %for k = 1:numel(row)
            %    tmp(row(k) - min_row + 1, col(k) - min(col) + 1) = true;
            %end
            %is_rectangle = all(tmp(:));
            
            % method 2
            num_cell_horz = (max_col - min_col + 1);
            num_cell_vert = (max_row - min_row + 1);
            is_rectangle = (numel(row) == (num_cell_horz * num_cell_vert));
                
            if is_rectangle
                left = sum(dx(1:min_col - 1));
                right = sum(dx(1:max_col));
                upper = sum(dz(1:min_row - 1));
                bottom = sum(dz(1:max_row));
                width = right - left;
                height = bottom - upper;
                x_mid = left + width / 2;
                y_mid = upper + height / 2;
                tmp_para{j} = [right, upper, left, bottom, ...
                               x_mid, y_mid, width, height, rho_vals(j)];
                % visualize block position
                tmp_block_heatmap = zeros(nz, nx);
                for k = 1:numel(row)
                    tmp_block_heatmap(row(k), col(k)) = 1;
                end
                block_heatmap = block_heatmap + tmp_block_heatmap;
                
            else
                tmp_para{j} = [nan, nan, nan, nan, ...
                               nan, nan, nan, nan, rho_vals(j)];
            end
        end
        tmp_paras{i} = cell2mat(tmp_para);
    end
    block_para{n} = cell2mat(tmp_paras);
end

block_para = array2table(cell2mat(block_para), ...
                          'VariableNames', ...
                          {'right', 'upper', 'left', 'bottom', ...
                           'x_mid', 'y_mid', 'width', 'height', 'rho'});
