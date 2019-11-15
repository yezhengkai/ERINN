import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib

from erinn.python.utils.io_utils import read_pkl, read_config_file

#%% setting
# io
train_dir = '../data/raw_data/train'
test_dir = '../data/raw_data/train'
# random forest
num_train = 1000
num_test = 100
num_cpu = os.cpu_count()
num_tree = 3
random_seed = 42
# plot
config = read_config_file('../config/config.yml')
nx = config['nx']
nz = config['nz']
limit = 5

#%% Read data
X_train = np.array([], dtype='float64')
y_train = np.array([], dtype='float64')
X_test = np.array([], dtype='float64')
y_test = np.array([], dtype='float64')

for i in range(num_train):
    filename = os.path.join(train_dir, f'raw_data_{i+1}.pkl')
    print(filename)
    data = read_pkl(filename)
    X_train = np.vstack((X_train, data['inputs'])) if X_train.size else data['inputs']
    y_train = np.vstack((y_train, data['targets'])) if y_train.size else data['targets']

for i in range(num_train, num_train + num_test):
    filename = os.path.join(train_dir, f'raw_data_{i+1}.pkl')
    print(filename)
    data = read_pkl(filename)
    X_test = np.vstack((X_test, data['inputs'])) if X_test.size else data['inputs']
    y_test = np.vstack((y_test, data['targets'])) if y_test.size else data['targets']


# Random Forest
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators=num_tree, n_jobs=num_cpu, random_state=random_seed)
# Train the model on training data
rf.fit(X_train, y_train)

print('R\N{SUPERSCRIPT TWO}:',  rf.score(X_test, y_test))
y_pred = rf.predict(X_test)

for i in range(y_pred.shape[0]):
    plt.imshow(y_pred[i, :].reshape(nz, nx))
    plt.colorbar()
    plt.show()
    plt.imshow(y_test[i, :].reshape(nz, nx))
    plt.colorbar()
    plt.show()
    if i == limit:
        break


#%% save model
os.makedirs('../models/random_forest', exist_ok=True)
joblib.dump(rf, '../models/random_forest/rf.pkl')
