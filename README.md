# ERINN
Electrical resistivity imaging based on deep learning.

ERINN contains convenient functions for using deep learning to solve ERI problems.

The result can then provide a quick view of the resistivity and an useable initial model in the inversion routine.

Here are the specific steps:
1. Generate data using the python package 'simpeg'. (simpeg has not been embedded)
2. Preprocess the data and split it into training/validation/testing.
3. Train the neural network.
4. Verify the quality of the prediction with some figures and metrics.

---
# Warning
This version is unstable. Do not use it now.

Since the package is under construction, API will change frequently.

(Note that we will discard the matlab part and rewrite the entire package in python.)

Please star us for upcoming updates!

---
## Requirement
- python >= 3.6
- matplotlib >= 3.0.3
- numpy >= 1.16.2
- numba >= 0.43.0
- ruamel.yaml >= 0.16.5
- simpeg >= 0.13.0 (development state)
- tensorflow = 2.0.0

---
## ERI project
Under the ERI directory, there is a template [template_python](ERI/template_python).

You can use the template directly and change the name of the template to a meaningful name, 
such as a field name or an experiment name.

This template has some useful scripts:
1. [generate_dataset.py](ERI/template_python/scripts/generate_dataset.py)
    - Generate synthetic resistivity data and resistance(potential difference divided by current)
      receive by electrode array.
2. [preprocess_dataset.py](ERI/template_python/scripts/preprocess_dataset.py)
    - Preprocess the dataset (such as adding noise) and save it to a new directory.
3. [training.py](ERI/template_python/scripts/training.py)
    - Train the neural network.
4. [predict_resistivity.py](ERI/template_python/scripts/predict_resistivity.py)
    - Predict the underground resistivity.
5. [predict_potential_over_current.py](ERI/template_python/scripts/predict_potential_over_current.py)
    - Use predictive resistivity to simulate potential and get resistance.
6. [plot_subsurface.py](ERI/template_python/scripts/plot_subsurface.py)
    - Plot some figures to check the robustness of neural network.

Some parameters in above scripts are controlled by [config.yml](ERI/template_python/config/config.yml).

---
# Scheduled milestone

- [x] Generate synthetic resistivity model randomly.
    - [x] Control background, embedded rectangles and embedded circles resistivity (value and geometry).
    - [x] Save data as pickle files.
    - [x] Progress bar.
    - [x] Parallel version.
- [ ] Training neural network.
    - [x] Data generator. Read the pickle file and provide the data to the neural network in an appropriate form.
    - [x] Data augmentation in data generator.
    - [ ] Allows users to import custom models and callbacks written in python files via configuration file.
    - [x] Save weights of neural network.
- [ ] Predict resistivity.
    - [x] Save data as pickle files.
    - [ ] Progress bar.
    - [ ] Parallel version.
- [ ] Predict resistance (potential difference divided by current)
    - [x] Save data as pickle files.
    - [ ] Progress bar.
    - [ ] Parallel version.
- [ ] Plot resistivity.
    - [x] synthetic data
        - [x] crossplot
        - [ ] contour
        - [X] heatmap
    - [ ] filed data
        - [ ] crossplot
        - [ ] contour
    - [ ] Parallel version
    - [ ] Make figures more aesthetic.
- [ ] Go to tensorflow 2.0
- [ ] Change forward modeling code from FW2_5D to simpeg. (we can use terrain for forward modeling.)
- [ ] Build docker image. (optional)
- [ ] Publishing to PyPi. (optional)
- [ ] GUI (optional)