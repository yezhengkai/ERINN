function data = load_daily_data(varargin)

% default parameters for keyword arguments
default_Start = 'None'; %datestr(now, 'yyyymmdd');
default_End = 'None'; %datestr(now, 'yyyymmdd');

% parse input arguments
p = inputParser;
h5_expr = '\.(h5|hdf5)\>';
valid_file = @(x) regexp(x, h5_expr) && exist(x,'file');
valid_datestr = @(x) strcmp(num2str(x), 'None') ||...
                    isdatetime(datetime(num2str(x),...
                                'InputFormat', 'yyyyMMdd'));
addRequired(p, 'h5', valid_file);
addParameter(p, 'start', default_Start, valid_datestr);
addParameter(p, 'end', default_End, valid_datestr);
parse(p, varargin{:});

% set variables
h5 = p.Results.h5;
% calc_date = p.Results.date;
if ~strcmp(num2str(p.Results.start), 'None') &&...
        ~strcmp(num2str(p.Results.end), 'None')
    Start = datenum(num2str(p.Results.start), 'yyyymmdd');
    End = datenum(num2str(p.Results.end), 'yyyyMMdd');
elseif strcmp(num2str(p.Results.start), 'None') &&...
        ~strcmp(num2str(p.Results.end), 'None')
    Start = datenum('00000101', 'yyyymmdd');
    End = datenum(num2str(p.Results.end), 'yyyymmdd');
elseif ~strcmp(num2str(p.Results.start), 'None') &&...
        strcmp(num2str(p.Results.end), 'None')
    Start = datenum(num2str(p.Results.start), 'yyyymmdd');
    End = datenum('99991231', 'yyyymmdd');
else
    Start = default_Start;
    End = default_End;
end

data_info = h5info(h5, '/daily_data');
data = table();
date_expr = '(?<date>\d{4}\d{2}\d{2})';
if strcmp(Start, 'None') &&...
        strcmp(End, 'None')
    for i = 1:numel(data_info.Groups)
        token = regexp(data_info.Groups(i).Name, date_expr, 'names');
        date = cellstr(token.date);
        date_group = data_info.Groups(i).Name;
        try
            pred_V = h5read(h5, strcat(date_group, '/pred_V'));
            obs_V = h5read(h5, strcat(date_group, '/obs_V'));
            pred_log_rho = h5read(h5, strcat(date_group, '/pred_log_rho'));
            data = [data; table(date, obs_V, pred_V, pred_log_rho)];
        catch
            obs_V = h5read(h5, strcat(date_group, '/obs_V'));
            pred_log_rho = h5read(h5, strcat(date_group, '/pred_log_rho'));
            data = [data; table(date, obs_V, pred_log_rho)];
        end
    end
else
    for i = 1:numel(data_info.Groups)
        token = regexp(data_info.Groups(i).Name, date_expr, 'names');
        date = datenum(token.date, 'yyyymmdd');
        if date >= Start && date <= End
            date_group = data_info.Groups(i).Name;
            try
                pred_V = h5read(h5, strcat(date_group, '/pred_V'));
                obs_V = h5read(h5, strcat(date_group, '/obs_V'));
                pred_log_rho = h5read(h5, strcat(date_group, '/pred_log_rho'));
                data = [data; table(date, obs_V, pred_V, pred_log_rho)];
            catch
                obs_V = h5read(h5, strcat(date_group, '/obs_V'));
                pred_log_rho = h5read(h5, strcat(date_group, '/pred_log_rho'));
                data = [data; table(date, obs_V, pred_log_rho)];
            end
        end
    end
end

if ~size(data)
    fprintf(['There is no dataset in HDF5, ',...
        'or there is no dataset in a specific time span.\n']);
    return;
end

end