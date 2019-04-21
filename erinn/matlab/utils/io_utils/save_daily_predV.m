function save_daily_predV(varargin)

% parse input arguments
p = inputParser;
h5_expr = '\.(h5|hdf5)\>';
valid_file = @(x) regexp(x, h5_expr) && exist(x,'file');
valid_data = @(x) istable(x);
addRequired(p, 'h5', valid_file);
addRequired(p, 'daily_data', valid_data);
parse(p, varargin{:});

h5 = p.Results.h5;
pred_V = p.Results.daily_data.pred_V;
date_cell = p.Results.daily_data.date;

for i = 1:length(date_cell)
    if i == 1
        dset_info = h5info(h5, strcat('/daily_data/',...
                                       date_cell{1},...
                                       '/obs_V'));
        chunk_size = dset_info.ChunkSize;
        dset_size = dset_info.Dataspace.Size;
    end
    dset_name = strcat('/daily_data/', date_cell{i}, '/pred_V');
    
    try
        h5create(h5, dset_name, dset_size, 'ChunkSize', chunk_size);
        h5write(h5, dset_name, pred_V(i, :));
    catch
        h5write(h5, dset_name, pred_V(i, :));
    end
    
end
