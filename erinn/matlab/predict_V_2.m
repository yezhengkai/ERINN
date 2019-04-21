function predict_V_2(infile, varargin)
% Read HDF5 file and calculate predicted electrical potential.
%
% Parameters
% -------------
% infile: string
%     hdf5 file for simulation .
% outfile: string
%     hdf5 file for saving simulation result.
%     outfile equals infile if user didn't give outfile.
% date: kwarg
%     value: {'all', 'specify'}
%     Choose time range for simulation.
% Start: kwarg
%     value: string, format is 'yyyymmdd'.
% End: kwarg
%     value: string, format is 'yyyymmdd'.
% test_type: kwarg
%     value: {'None', 'all', 'Random_drop', 'C_drop',
%             'P_drop', 'R_C', 'R_P', 'C_P'}.
%     Test the generalization of neural network.


% check infile is exist or not
if ~exist(infile,'file')
    error('Input file does not exist.');
end

% outfile equals infile if user didn't give outfile
default_outfile = infile;
% calculate delta V for all dates in the h5 file
default_date = 'all';
expected_date = {'all', 'specify'};
default_Start = 'None'; %datestr(now, 'yyyymmdd');
default_End = 'None'; %datestr(now, 'yyyymmdd');
default_test = 'None';
expected_test = {'None', 'all', 'Random_drop', 'C_drop', 'P_drop',...
                      'R_C', 'R_P', 'C_P'};

% parse input arguments
p = inputParser;
h5_expr = '\.(h5|hdf5)\>';
valid_file = @(x) regexp(x, h5_expr);
valid_datestr = @(x) strcmp(num2str(x), 'None') ||...
                    isdatetime(datetime(num2str(x),...
                                'InputFormat', 'yyyyMMdd'))                            ;
addRequired(p, 'infile', valid_file);
addOptional(p, 'outfile', default_outfile, valid_file);
addParameter(p, 'date', default_date,...
    @(x) any(validatestring(x, expected_date)));
addParameter(p, 'Start', default_Start, valid_datestr);
addParameter(p, 'End', default_End, valid_datestr);
addParameter(p, 'test_type', default_test,...
    @(x) any(validatestring(x, expected_test)));
parse(p, infile, varargin{:});

% set variable
outfile = p.Results.outfile;
% if ~strcmp(infile, outfile) 
if ~exist(outfile, 'file')
    copyfile(infile, outfile);
end
calc_date = p.Results.date;
if ~strcmp(num2str(p.Results.Start), 'None') &&...
        ~strcmp(num2str(p.Results.End), 'None')
    Start = datenum(num2str(p.Results.Start), 'yyyymmdd');
    End = datenum(num2str(p.Results.End), 'yyyyMMdd');
elseif strcmp(num2str(p.Results.Start), 'None') &&...
        ~strcmp(num2str(p.Results.End), 'None')
    Start = datenum('00000101', 'yyyymmdd');
    End = datenum(num2str(p.Results.End), 'yyyymmdd');
elseif ~strcmp(num2str(p.Results.Start), 'None') &&...
        strcmp(num2str(p.Results.End), 'None')
    Start = datenum(num2str(p.Results.Start), 'yyyymmdd');
    End = datenum('99991231', 'yyyymmdd');
else
    Start = default_Start;
    End = default_End;
end
test_type = p.Results.test_type;

% read variables for simulation
dx = h5read(infile, '/glob_para/dx');
dz = h5read(infile, '/glob_para/dz');
srcloc = h5read(infile, '/glob_para/srcloc');
recloc = h5read(infile, '/glob_para/recloc');
srcnum = h5read(infile, '/glob_para/srcnum');
m = matfile('Q.mat');
Q = m.Q;

data_info = h5info(infile, '/daily_data');
dset_list = get_dset_name(calc_date, data_info, Start, End, test_type);
dset_list_len = numel(dset_list);
if dset_list_len == 0
    fprintf(['There is no dataset in HDF5, ',...
        'or there is no dataset in a specific time span.\n']);
    return;
end


pred_S = cell(1, dset_list_len);
pred_V = cell(1, dset_list_len);
% Avoid using parfor IO because it may cause errors...
% But this method will increase memory usage.
for i = 1:dset_list_len
    dset = dset_list{i};
    fprintf('Read %s\n', dset);
    pred_log_rho = h5read(infile, dset);
    pred_log_rho = cast(pred_log_rho, 'double');
    % convert to sigma(conductivity).
    pred_S{i} = 1./(power(10, pred_log_rho));
end

parfor i = 1:dset_list_len
    
    dset = dset_list{i};
%     fprintf('Number of dset: %d\n', i);
    fprintf('Start simulation.\n');
    fprintf('Dataset: %s\n', dset);
%     pred_log_rho = h5read(infile, dset);
%     pred_log_rho = cast(pred_log_rho, 'double');
%     pred_S = 1./(power(10, pred_log_rho));
    s = reshape(pred_S{i}, numel(dx), numel(dz));
    [Para] = get_2_5Dpara_noQ(srcloc, dx, dz, s, 4, recloc, srcnum);
    Para.Q = Q;
    [dobs, U] = dcfw2_5D(s, Para);
    pred_V{i} = dobs;
end

% Avoid using parfor IO because it may cause errors..
% But this method will increase memory usage.
for i = 1:dset_list_len
    
    % default chunk size is [4371 1]
    % check if the pred_V size is larger than default chunk size
    dims = size(pred_V{i});
    chunk_size = min([4371 1], dims);
    dset = dset_list{i};
    dset1 = strrep(dset, 'pred_log_rho', 'pred_V');
    
    
    try
        h5create(outfile, dset1, dims,...
            'Datatype', 'double', 'Chunksize', chunk_size);
        msg = ['Create ', dset1, '\n'];
        fprintf(msg);
    catch
        msg = ['The dataset ', dset1, ' already exists.\n'];
        fprintf(msg);
    end
    h5write(outfile, dset1, pred_V{i});
    fprintf('Write %s\n', dset1);
    
end


end

function test_info_list = check_test_type(date_group, test_type)
% get test dataset path in hdf5

switch test_type
    case('None')
        test_info_list = [];
    case('Random_drop')
        test_info_list = h5info([date_group, '/Random_drop']);
    case('C_drop')
        test_info_list = h5info([date_group, '/C_drop']);
    case('P_drop')
        test_info_list = h5info([date_group, '/P_drop']);
    case('R_C')
        test_info_list(1) = h5info([date_group, '/Random_drop']);
        test_info_list(2) = h5info([date_group, '/C_drop']);
    case('R_P')
        test_info_list(1) = h5info([date_group, '/Random_drop']);
        test_info_list(2) = h5info([date_group, '/P_drop']);
    case('C_P')
        test_info_list(1) = h5info([date_group, '/C_drop']);
        test_info_list(2) = h5info([date_group, '/P_drop']);
    case('all')
        test_info_list(1) = h5info([date_group, '/Random_drop']);
        test_info_list(2) = h5info([date_group, '/C_drop']);
        test_info_list(3) = h5info([date_group, '/P_drop']);
end

end

function dset_list = get_dset_name(calc_date, data_info,...
                                   Start, End, test_type)
% get all dataset path of hdf5 and return it as a array

dset_list = cell(0);
if strcmp(calc_date, 'all') && strcmp(Start, 'None') &&...
        strcmp(End, 'None')
    for i = 1:numel(data_info.Groups)
        date_group = data_info.Groups(i).Name;
        dset_list(end+1) = cellstr(strcat(date_group, '/pred_log_rho'));
        test_info_list = check_test_type(date_group, test_type);
        for test_info = test_info_list
            for k = 1:numel(test_info.Groups)
                dset_list(end+1) = cellstr(strcat(test_info.Groups(k).Name,...
                                   '/pred_log_rho'));
            end
        end
    end
else
    date_expr = '(?<date>\d{4}\d{2}\d{2})';
    for i = 1:numel(data_info.Groups)
        token = regexp(data_info.Groups(i).Name, date_expr, 'names');
        date = datenum(token.date, 'yyyymmdd');
        if date >= Start && date <= End
            date_group = data_info.Groups(i).Name;
            dset_list(end+1) = cellstr(strcat(date_group, '/pred_log_rho'));
            test_info_list = check_test_type(date_group, test_type);
            for test_info = test_info_list
                for k = 1:numel(test_info.Groups)
                    dset_list(end+1) = cellstr(strcat(test_info.Groups(k).Name,...
                                   '/pred_log_rho'));
                end
            end
        end
    end
end

end