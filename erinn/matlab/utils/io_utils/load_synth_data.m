function data = load_synth_data(varargin)
% Load synthetic data
%
% Parameters
% ----------
%
%
% Returns
% -------
% data : sturct
%     Structure with 'synth_V', 'pred_V', 'synth_log_rho', 'pred_log_rho'
%     fieldnames.

% parse input arguments
p = inputParser;
h5_expr = '\.(h5|hdf5)\>';
default_index = 'all';
valid_file = @(x) regexp(x, h5_expr) && exist(x,'file');
valid_index = @(x) isvector(x) & isnumeric(x);
addRequired(p, 'h5', valid_file);
% addRequired(p, 'index', valid_index);
addParameter(p, 'index', default_index, valid_index);
parse(p, varargin{:});

h5 = p.Results.h5;
if ~strcmp(p.Results.index, 'all')
    samples_index = unique(p.Results.index);
    if find(diff(samples_index) ~= 1)
        error('Index must be continuous.');
    end
end

data_info = h5info(h5);
% check if the h5 file has the '/synth_data' group
j = 0;
for i = 1:length(data_info.Groups)
    if strcmp(data_info.Groups(i).Name, '/synth_data')
        j = i;
    end
end

if j
    data.synth_V = [];
    data.pred_V = [];
    data.synth_log_rho = [];
    data.pred_log_rho = [];
    if strcmp(p.Results.index, 'all') % load whole matrix
        % walk through all dataset in '/synth_data' group
        for i = 1:length(data_info.Groups(j).Datasets)
            dset = data_info.Groups(j).Datasets(i);
            switch dset.Name
                case 'synth_V'
                    data.synth_V = h5read(h5, '/synth_data/synth_V');
                    fprintf('load all /synth_data/synth_V\n');
                case 'pred_V'
                    data.pred_V = h5read(h5, '/synth_data/pred_V');
                    fprintf('load all /synth_data/pred_V\n');
                case 'synth_log_rho'
                    data.synth_log_rho = h5read(h5, '/synth_data/synth_log_rho');
                    fprintf('load all /synth_data/synth_log_rho\n');
                case 'pred_log_rho'
                    data.pred_log_rho = h5read(h5, '/synth_data/pred_log_rho');
                    fprintf('load all /synth_data/pred_log_rho\n');
            end
        end
    else
        % load from specific index
        base_row = samples_index(1);
        % number of rows need to be load
        nb_row = samples_index(end) - samples_index(1) + 1;
        % walk through all dataset in '/synth_data' group
        for i = 1:length(data_info.Groups(j).Datasets)
            dset = data_info.Groups(j).Datasets(i);
            switch dset.Name
                case 'synth_V'
                    data_size = dset.Dataspace.Size;
                    data.synth_V = h5read(h5, '/synth_data/synth_V',...
                        [base_row, 1], [nb_row, data_size(2)]);
                    fprintf('load /synth_data/synth_V(%d:%d, :)\n',...
                            samples_index(1), samples_index(end));
                case 'pred_V'
                    data_size = dset.Dataspace.Size;
                    data.pred_V = h5read(h5, '/synth_data/pred_V',...
                        [base_row, 1], [nb_row, data_size(2)]);
                    fprintf('load /synth_data/pred_V(%d:%d, :)\n',...
                            samples_index(1), samples_index(end));
                case 'synth_log_rho'
                    data_size = dset.Dataspace.Size;
                    data.synth_log_rho = h5read(h5, '/synth_data/synth_log_rho',...
                        [base_row, 1], [nb_row, data_size(2)]);
                    fprintf('load /synth_data/synth_log_rho(%d:%d, :)\n',...
                            samples_index(1), samples_index(end));
                case 'pred_log_rho'
                    data_size = dset.Dataspace.Size;
                    data.pred_log_rho = h5read(h5, '/synth_data/pred_log_rho',...
                        [base_row, 1], [nb_row, data_size(2)]);
                    fprintf('load /synth_data/pred_log_rho(%d:%d, :)\n',...
                            samples_index(1), samples_index(end));
            end
        end
    end
else
    error('Can not find /synth_data group.');
end

end