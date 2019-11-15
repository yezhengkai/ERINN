if __name__ == '__main__':
    import os

    from erinn.python.preprocessing import make_processed_dataset

    # setting
    config_file = os.path.join('..', 'config', 'config.yml')

    # save processed data as pickle files
    # input: V/I (linear scale); targets: resistivity (log10 scale)
    make_processed_dataset(config_file)
