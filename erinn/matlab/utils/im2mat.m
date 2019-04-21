function X_scaled = im2mat(img, new_minmax, output_mat)
% This function is not general enough and should be discarded.

[filepath, name, ~] = fileparts(output_mat);
if ~exist(filepath, 'dir')
    mkdir(filepath);
end

X = double(rgb2gray(imread(img)));
new_minmax = sort(new_minmax);
newmin = new_minmax(1);
newmax = new_minmax(2);
oldmin = min(min(X));
oldmax = max(max(X));
X_std = (X - oldmin) / (oldmax - oldmin);
eval([name, ' = X_std * (newmax - newmin) + newmin;']);
X_scaled = X_std * (newmax - newmin) + newmin;
% imagesc(X_scaled);

eval(['save(output_mat, ''', name, ''', ''-v7.3'')']);

end

function X_scaled = minmax_scale(X, newmin, newmax)

oldmin = min(min(X));
oldmax = max(max(X));
X_std = (X - oldmin) / (oldmax - oldmin);
X_scaled = X_std * (newmax - newmin) + newmin;

end