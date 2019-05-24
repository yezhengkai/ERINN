function save_synth_predV(varargin)
% Save synthetic predictive \DeltaV/I to hdf5.
%
% Parameters
% ----------
% varargin : cell
%
% Returns
% -------
% None

% parse input arguments
p = inputParser;
h5_expr = '\.(h5|hdf5)\>';
default_index = 'append';
valid_file = @(x) regexp(x, h5_expr) && exist(x,'file');
valid_data = @(x) ismatrix(x);
valid_index = @(x) isvector(x) & isnumeric(x);
addRequired(p, 'h5', valid_file);
addRequired(p, 'pred_V', valid_data);
addParameter(p, 'index', default_index, valid_index);
parse(p, varargin{:});

h5 = p.Results.h5;
pred_V = p.Results.pred_V;
if ~strcmp(p.Results.index, 'append')
    samples_index = unique(p.Results.index);
    if find(diff(samples_index) ~= 1)
        error('Index must be continuous.');
    end
end

dset_info = h5info(h5, '/synth_data/synth_V');
chunk_size = dset_info.ChunkSize;
dset_size = dset_info.Dataspace.Size;
predV_size = size(pred_V);
assert(isequal(predV_size(2), dset_size(2)),...
        'The number of features in synth_V and pred_V must be the same.')
try
    h5create(h5, '/synth_data/pred_V', [Inf dset_size(2)],...
        'ChunkSize', chunk_size, 'FillValue', nan);
    if strcmp(p.Results.index, 'append')
        h5write(h5, '/synth_data/pred_V', pred_V, [1, 1], predV_size);
    else
        % write from specific index
        base_row = samples_index(1);
        % number of rows need to be writed
        num_row = samples_index(end) - samples_index(1) + 1;
        assert(isequal(predV_size, [num_row, dset_size(2)]),...
            'Number of Indices must be the same as the row number of pred_V.')
        h5write(h5, '/synth_data/pred_V', pred_V, [base_row, 1],...
                [num_row, dset_size(2)]);
    end
catch
    dset_info = h5info(h5, '/synth_data/pred_V');
    dset_size = dset_info.Dataspace.Size;
    if strcmp(p.Results.index, 'append')
        h5write(h5, '/synth_data/pred_V', pred_V, [dset_size(1) + 1, 1],...
            predV_size);
    else
        % write from specific index
        base_row = samples_index(1);
        % number of rows need to be writed
        num_row = samples_index(end) - samples_index(1) + 1;
        assert(isequal(predV_size, [num_row, dset_size(2)]),...
            'Number of Indices must be the same as the row number of pred_V.')
        h5write(h5, '/synth_data/pred_V', pred_V, [base_row, 1],...
                [num_row, dset_size(2)]);
    end
end
