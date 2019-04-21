function write_urf(output_urf, Tx_id, Rx_id, RxP2_id, coord, data)
% Write urf file
% 
% Parameters
% ----------
% output_urf : str
% 
% Tx_id : double
% 
% Rx_id : double
% 
% RxP2_id : double
% 
% coord : double
% 
% data : double
% 

if exist(output_urf, 'file')
    warning('urf file exist. Discard existing contents');
end

%
fid = fopen(output_urf, 'wt');

fprintf(fid, 'Tx\n');
fprintf(fid, '%d,', Tx_id(1:end-1));
fprintf(fid, '%d\n', Tx_id(end));

fprintf(fid, 'Rx\n');
fprintf(fid, '%d,', Rx_id(1:end-1));
fprintf(fid, '%d\n', Rx_id(end));

fprintf(fid, 'RxP2\n');
if isnan(RxP2_id)
    fprintf(fid, '\n');
else
    fprintf(fid, '%d\n', RxP2_id);
end

fprintf(fid, '\n');
fprintf(fid, ':Geometry\n');
fprintf(fid,'%d,%0.2f,%0.2f,%0.2f\n', coord');

fprintf(fid, '\n');
fprintf(fid, ':Measurements\n');
fprintf(fid,'%d,%d,%d,%d,%.8f,%d,%d\n', data');

fclose(fid);

fprintf('Write %s\n', output_urf);