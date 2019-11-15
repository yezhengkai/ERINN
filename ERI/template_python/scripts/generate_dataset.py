if __name__ == '__main__':
    import os

    from erinn.python.FW2_5D.fw2_5d_ext import make_dataset

    # setting
    config_file = os.path.join('..', 'config', 'config.yml')

    # save raw data as pickle files
    # input: V/I (linear scale); targets: resistivity (log10 scale)
    make_dataset(config_file)
