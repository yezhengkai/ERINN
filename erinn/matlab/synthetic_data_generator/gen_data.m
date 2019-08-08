function gen_data(json)
%
% use 1m*1m square mesh!
% Except read_block, other cores didn't add to package.



% check json is exist or not
if ~exist(json, 'file')
    error('json file does not exist.');
end
% parse json
para = jsondecode(fileread(json));

% get current pool
pool = gcp('nocreate'); 
if isempty(pool) % If pool is empty, create one.
    % create parallel pool (parpool) using the 'local' profile
    numcores = feature('numcores');
    parpool('local', numcores);
end


t_start = clock;  % timeit
fpath = para.output_path; %'.\Training_Dataset';
if ~exist(fpath, 'dir')
    mkdir(fpath);
end
basename = para.generator.core; %'random_block';

%% generate conductivity structure
% para.generator.nx = para.nx;
% para.generator.nz = para.nz;
switch para.generator.core
    case 'rand_block'
        Targets = rand_block(para);
    case 'rand_layer' 
        Targets = rand_layer(para);
    case 'rand_column'
        Targets = rand_column(para);
    case 'rand_field'
        Targets = rand_field(para);
end

%% forward simulation
% para.simulate.nx = para.nx;
% para.simulate.nz = para.nz;
simu_para = fwd_simu_para(para);
Inputs = fwd_simu(Targets, simu_para);


%% save dataset as HDF5
formatOut = 'yyyymmdd';
t = datestr(now, formatOut);
fname = [basename, '_', t, '_0.h5'];
h5 = fullfile(fpath, fname);

% if file exist, increase index
ind = 1;
while exist(h5, 'file')
    fname = [basename, '_', t, '_', num2str(ind), '.h5'];
    h5 = fullfile(fpath, fname);
    ind = ind + 1;
end

save_data_h5(h5, Inputs, Targets, simu_para);
%========================
% % create hdf5
% % check if the size is larger than default chunk size
% chunk_size = min([125 4000], size(Inputs));
% h5create(fulname, '/Inputs', size(Inputs),...
%          'Datatype', 'double', 'Chunksize', chunk_size);
% chunk_size = min([125 400], size(Targets));
% h5create(fulname, '/Targets', size(Targets),...
%          'Datatype', 'double', 'Chunksize', chunk_size);
% chunk_size = min([100 4], size(simu_para.srcloc));
% h5create(fulname,'/srcloc',size(simu_para.srcloc),...
%          'Datatype','double','Chunksize',chunk_size);
% h5create(fulname, '/dx', size(simu_para.dx), 'Datatype', 'double');
% h5create(fulname, '/dz', size(simu_para.dz), 'Datatype', 'double');
% chunk_size = min([400 4], size(simu_para.recloc));
% h5create(fulname, '/recloc', size(simu_para.recloc),...
%          'Datatype', 'double', 'Chunksize', chunk_size);
% chunk_size = min([400 1], size(simu_para.srcnum));
% h5create(fulname,'/srcnum',size(simu_para.srcnum),...
%          'Datatype','double','Chunksize',chunk_size);
% % write hdf5
% h5write(fulname,'/Inputs',Inputs);
% h5write(fulname,'/Targets',Targets);
% h5write(fulname,'/srcloc', simu_para.srcloc);
% h5write(fulname,'/dx', simu_para.dx);
% h5write(fulname,'/dz', simu_para.dz);
% h5write(fulname,'/recloc', simu_para.recloc);
% h5write(fulname,'/srcnum', simu_para.srcnum);

% %% save dataset as mat file
% fulname = [fulname(1:end-2), 'mat'];
% save(fulname,'Inputs', 'Targets', '-v7.3'); 
% save(fulname,'-struct','simu_para',...
%     'srcloc','dx','dz','recloc','srcnum','-append');
%========================
%% 
t_elapsed = etime(clock,t_start)/60;
fprintf('Elapsed time is %f minutes.\n',t_elapsed);