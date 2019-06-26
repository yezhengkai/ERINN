function gen_glob_para_h5(h5, config_json)

% check h5 is exist or not
if exist(h5, 'file')
    warning('h5 file already exists. Delete it.');
    delete(h5);
end

% check json is exist or not
if ~exist(config_json, 'file')
    error('json file does not exist.');
end
% parse json
para = jsondecode(fileread(config_json));

urf = para.simulate.geometry_urf;
[Tx_id, Rx_id, RxP2_id, coord, ~] = read_urf(urf);
[srcloc, dx, dz, recloc, srcnum] = prepare_for_get_2_5Dpara(para);
nx = para.nx;
nz = para.nz;

array_len = max(coord(:, 2)) - min(coord(:, 2));
coord_in_model = coord - [0, array_len/2, 0, 0];

%% create hdf5
h5create(h5, '/glob_para/Tx_id', size(Tx_id), 'Datatype', 'double');
h5create(h5, '/glob_para/Rx_id', size(Rx_id), 'Datatype', 'double');
h5create(h5, '/glob_para/RxP2_id', size(RxP2_id), 'Datatype', 'double');
h5create(h5, '/glob_para/coord', size(coord), 'Datatype', 'double');
h5create(h5, '/glob_para/coord_in_model', size(coord_in_model), ...
         'Datatype', 'double');

chunk_size = min([100 4], size(srcloc));
h5create(h5,'/glob_para/srcloc', size(srcloc), ...
         'Datatype', 'double', 'Chunksize', chunk_size);
h5create(h5, '/glob_para/dx', size(dx), 'Datatype', 'double');
h5create(h5, '/glob_para/dz', size(dz), 'Datatype', 'double');
chunk_size = min([400 4], size(recloc));
h5create(h5, '/glob_para/recloc', size(recloc), ...
         'Datatype', 'double', 'Chunksize', chunk_size);
chunk_size = min([400 1], size(srcnum));
h5create(h5, '/glob_para/srcnum',size(srcnum), ...
         'Datatype', 'double', 'Chunksize', chunk_size);
		 
h5create(h5, '/glob_para/nx', size(nx), 'Datatype', 'double');
h5create(h5, '/glob_para/nz', size(nz), 'Datatype', 'double');

%% write hdf5
% h5write(filename,datasetname,data,start,count,stride)
h5write(h5, '/glob_para/Tx_id', Tx_id);
h5write(h5, '/glob_para/Rx_id', Rx_id);
h5write(h5, '/glob_para/RxP2_id', RxP2_id);
h5write(h5, '/glob_para/coord', coord);
h5write(h5, '/glob_para/coord_in_model', coord_in_model);
h5write(h5, '/glob_para/srcloc', srcloc);
h5write(h5, '/glob_para/dx', dx);
h5write(h5, '/glob_para/dz', dz);
h5write(h5, '/glob_para/recloc', recloc);
h5write(h5, '/glob_para/srcnum', srcnum);
h5write(h5, '/glob_para/nx', nx);
h5write(h5, '/glob_para/nz', nz);

fprintf('Create %s.\n', h5);
