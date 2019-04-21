function save_data_h5(h5, Inputs, Targets, simu_para, varargin)
% TODO: add this functionality? -> if .hdf5(.mat) exist, append it

% default value
default_optiton = false;
% parse input arguments
p = inputParser;
addRequired(p, 'h5');
addRequired(p, 'Inputs');
addRequired(p, 'Targets');
addRequired(p, 'simu_para');
addParameter(p, 'save_mat', default_optiton, @(x) islogical(x));
parse(p, h5, Inputs, Targets, simu_para, varargin{:});

% assign parse results
h5 = p.Results.h5;
Inputs = p.Results.Inputs;
Targets = p.Results.Targets;
simu_para = p.Results.simu_para;
save_mat = p.Results.save_mat;

% create hdf5
% check if the size is larger than default chunk size
chunk_size = min([125 4000], size(Inputs));
h5create(h5, '/Inputs', [Inf, size(Inputs, 2)],...
         'Datatype', 'double', 'Chunksize', chunk_size);
chunk_size = min([125 400], size(Targets));
h5create(h5, '/Targets', [Inf, size(Targets, 2)],...
         'Datatype', 'double', 'Chunksize', chunk_size);
chunk_size = min([100 4], size(simu_para.srcloc));
h5create(h5,'/srcloc',size(simu_para.srcloc),...
         'Datatype','double','Chunksize',chunk_size);
h5create(h5, '/dx', size(simu_para.dx), 'Datatype', 'double');
h5create(h5, '/dz', size(simu_para.dz), 'Datatype', 'double');
chunk_size = min([400 4], size(simu_para.recloc));
h5create(h5, '/recloc', size(simu_para.recloc),...
         'Datatype', 'double', 'Chunksize', chunk_size);
chunk_size = min([400 1], size(simu_para.srcnum));
h5create(h5,'/srcnum',size(simu_para.srcnum),...
         'Datatype','double','Chunksize',chunk_size);
% write hdf5
% h5write(filename,datasetname,data,start,count,stride)
h5write(h5, '/Inputs', Inputs, [1, 1], size(Inputs));
h5write(h5, '/Targets', Targets, [1, 1], size(Targets));
h5write(h5, '/srcloc', simu_para.srcloc);
h5write(h5, '/dx', simu_para.dx);
h5write(h5, '/dz', simu_para.dz);
h5write(h5, '/recloc', simu_para.recloc);
h5write(h5, '/srcnum', simu_para.srcnum);

fprintf('Create %s.\n', h5);

%% save dataset as mat file
if save_mat
    h5 = [h5(1:end-2), 'mat'];
    save(h5, 'Inputs', 'Targets', '-v7.3');
    save(h5, '-struct', 'simu_para',...
        'srcloc','dx','dz','recloc','srcnum', '-append');
    
    fprintf('Create %s.\n', h5);
end