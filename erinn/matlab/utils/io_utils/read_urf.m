function [Tx_id, Rx_id, RxP2_id, coord, data] = read_urf(urf)
%%
% check urf is exist or not
if ~exist(urf, 'file')
    error('urf file does not exist.');
end

% predefine output variables
Tx_id = [];
Rx_id = [];
RxP2_id = [];
coord = [];
data = [];

fid = fopen(urf, 'r');
while ~feof(fid)
    % read line from file, removing newline characters
    line = fgetl(fid);
    
    if strcmp(line, 'Tx')
        line = fgetl(fid);
        Tx_id = str2double(strsplit(line,','));
%         display(line)
    elseif strcmp(line, 'Rx')
        line = fgetl(fid);
        Rx_id = str2double(strsplit(line,','));
%         point_num = nchoosek(numel(Tx_num),2)*nchoosek(numel(Rx_num),2);
%         data = zeros(point_num, 7);
%         display(line)
    elseif strcmp(line, 'RxP2')
        line = fgetl(fid);
        RxP2_id = str2double(strsplit(line,','));
    elseif strfind(line, 'Geometry')
%         E_num = max(max(Tx_id), max(Rx_id));
        E_num = max([Tx_id, Rx_id, RxP2_id]);
%         Pos = zeros(E_num, 4);
        fmt = '%f,%f,%f,%f';
        size_tmp = [4, E_num];
        tmp = fscanf(fid, fmt, size_tmp);
        coord = tmp';
%         for i = 1:E_num
%             line = fgetl(fid);
%             xyz = str2double(strsplit(line,','));
%             Pos(i,:) = xyz;
%         end
    elseif strfind(line, 'Measurements')
        point_num = nchoosek(numel(Tx_id),2)*nchoosek(numel(Rx_id),2);
        fmt = '%f,%f,%f,%f,%f,%f,%f';
        size_tmp = [7, point_num];
        tmp = fscanf(fid, fmt, size_tmp);
        data = tmp';
%         data = zeros(point_num, 7);
%         for i = 1:point_num
%             line = fgetl(fid);
%             d = str2double(strsplit(line,','));
%             data(i,:) = d;
%         end
    end
%     display(line)
%     pause(0.5);
end
fclose(fid);

% save('para','Tx_num','Rx_num','Pos','data','-v7.3');