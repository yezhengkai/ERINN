from erinn.python.utils.io_utils import write_training_npz


if __name__ == "__main__":

    glob_para_h5 = '../config/glob_para.h5'

    # training data
    # change the path to suit your situation
    h5_list = ['../data/synthetic_data/rand_block_YYYYMMDD_0.h5', #將前面產生的h5檔輸入，可以寫個loop導入或是複製貼上
               '../data/synthetic_data/rand_block_YYYYMMDD_1.h5']
    npz_dir = '../data/processed_data/training'

    write_training_npz(glob_para_h5, h5_list, npz_dir, shape='3d')

    # testing data
    # change the path to suit your situation
    h5_list = ['../data/synthetic_data/rand_block_YYYYMMDD_2.h5']
    npz_dir = '../data/processed_data/testing'

    write_training_npz(glob_para_h5, h5_list, npz_dir, shape='3d')
