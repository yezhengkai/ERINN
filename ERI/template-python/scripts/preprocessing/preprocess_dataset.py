"""
Preprocess resistance (Delta V/I [ohm], linear scale)
and resistivity ([ohm*m], log10 scale).
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FILEDIR = os.path.dirname(__file__)

if __name__ == '__main__':
    from erinn.preprocessing import make_processed_dataset

    # setting
    config_file = os.path.join(FILEDIR, '..', '..', 'config', 'for_preprocess.yml')

    # Under "dataset_dir" defined in config_file,
    # the script will create a specific folder tree and
    # save the following data as pickle files within the corresponding folder.
    # resistance:
    #   add_noise => V/I with noise in linear scale.
    #                shape = (number of resistance,)
    #   log_transform => V/I in special log scale.
    #                    shape = (number of resistance,)
    #   to_midpoint => V/I in linear scale.
    #                  shape = (accumulated number of same midpoint,
    #                           number of midpoint)
    #                  or shape = (accumulated number of same midpoint,
    #                              number of midpoint, 1)
    #   to_txrx => V/I in linear scale.
    #              shape = (number of Tx pair, number of Rx pair)
    #              or shape = (number of Tx pair, number of Rx pair, 1)
    # resistivity:
    #   to_section => resistivity in log10 scale.
    #                 shape = (
    #                     number of cell center mesh in the z (y) direction,
    #                     number of cell center mesh in the x direction
    #                 )
    #                 or shape = (
    #                        number of cell center mesh in the z (y) direction,
    #                        number of cell center mesh in the x direction,
    #                        1
    #                     )
    # In addition, a simulator object containing information
    # about physical simulation will be saved.
    make_processed_dataset(config_file)
