#import numba
#numba.__version__#查看版本
#升級模組 $ pip install -U numba 
#安裝/更新 llvmlite $ conda install -c numba llvmlite 
from erinn.python.utils.io_utils import write_training_npz

if __name__ == "__main__":

    glob_para_h5 = '../config/glob_para.h5'

    # training data
    # change the path to suit your situation
    h5_list = ['../data/synthetic_data/rand_block_20190909_0.h5',
               '../data/synthetic_data/rand_block_20190909_1.h5',
               '../data/synthetic_data/rand_block_20190909_3.h5']
    npz_dir = '../data/processed_data/training'

    write_training_npz(glob_para_h5, h5_list, npz_dir, shape='3d')

    # testing data
    # change the path to suit your situation
    h5_list = ['../data/synthetic_data/rand_block_20190909_2.h5']
    npz_dir = '../data/processed_data/testing'

    write_training_npz(glob_para_h5, h5_list, npz_dir, shape='3d')
