"""
Generate resistance (Delta V/I [ohm], linear scale)
and resistivity ([ohm*m], log10 scale).
"""

# References
# ----------
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
FILEDIR = os.path.dirname(__file__)

if __name__ == '__main__':
    from erinn.simpeg_extended import make_dataset

    # setting
    config_file = os.path.join(FILEDIR, '..', '..', 'config', 'for_generate_data.yml')

    # Under "save_dataset_dir" defined in config_file,
    # the script will create a specific folder tree and
    # save the following data as pickle files within the corresponding folder.
    # resistance:
    #   V/I in linear scale. shape = (number of resistance,)
    # resistivity:
    #   resistivity in log10 scale. shape = (number of cell center mesh,)
    make_dataset(config_file)
