# ERINN
Electrical resistivity imaging based on deep learning.

ERINN contains convenient functions for using deep learning to solve ERI problems.

The result can then provide a quick view of the resistivity and an useable initial model in the inversion routine.

Here are the specific steps:
1. Generate synthetic data using the python package 'simpeg'.
2. Preprocess the data and split it into training/validation/testing.
3. Train the neural network.
4. Verify the quality of the prediction with some figures and metrics.

## How to run this project
First, you should download this repository and then follow the instructions below to install dependencies. Just choose a method you like.

### Use Anaconda
- Install [Anaconda](https://www.anaconda.com/products/individual) and set your preferred shell environment so that you can use the `conda` command.
- Open your preferred shell and change the directory to the repository you downloaded.
- Use `conda env update --prune --file environment.yml` to create a new conda environment and install dependencies. (If you want to create a new dev conda environment, use `conda env update --prune --file environment_dev.yml`)
- Use `conda develop src` to install package of the current project in "development mode".

### Use poetry
- Make sure you have the appropriate version of the python interpreter in your system.
- Install [poetry](https://python-poetry.org/docs/) and set your preferred shell environment so that you can use the `poetry` command.
- Open your preferred shell and change the directory to the repository you downloaded.
- Use `poetry install --no-dev` to install dependencies and package of the current project. (If you want to install dev dependencies, use `poetry install`)

### Use pip
- Make sure you have the appropriate version of the python interpreter and pip in your system.
- Open your preferred shell and change the directory to the repository you downloaded.
- Use `pip install -r requirement.txt` to install dependencies. (If you want to install dev dependencies, use `pip install -r requirement_dev.txt`)
- Use `pip install -e src` to install package of the current project in "development mode".


## ERI project
Under the ERI directory, there is a template [template-python](ERI/template-python).

You can use the template directly and change the name of the template to a meaningful name, 
such as a field name or an experiment name.

This template has some useful scripts:
1. [generate_dataset.py](ERI/template-python/scripts/generate-dataset/generate_dataset.py)
    - Generate synthetic resistivity data and resistance(potential difference divided by current)
      receive by electrode array.
2. [preprocess_dataset.py](ERI/template-python/scripts/preprocessing/preprocess_dataset.py)
    - Preprocess the dataset (such as adding noise) and save it to a new directory.
3. [training.py](ERI/template-python/scripts/training/training.py)
    - Train the neural network.
4. [predict_resistivity_log10.py](ERI/template-python/scripts/evaluation/predict_resistivity_log10.py)
    - Predict the underground resistivity.
5. [predict_resistance.py](ERI/template-python/scripts/evaluation/predict_resistance.py)
    - Use predictive resistivity to simulate potential and get resistance.
6. [plot_subsurface.py](ERI/template-python/scripts/visualization/plot_subsurface.py)
    - Plot some figures to check the robustness of neural network.

Some parameters in above scripts are controlled by yaml files in [config](ERI/template-python/config) directory.

---

## Warning
This version is unstable. Do not use it now.

Since the package is under construction, API will change frequently.

(Note that we will discard the matlab part and rewrite the entire package in python.)

Please star us for upcoming updates!

---

## Scheduled milestone

- [x] Generate synthetic resistivity model randomly.
    - [x] Control background, embedded rectangles and embedded circles resistivity (value and geometry).
    - [x] Save data as pickle files.
    - [x] Progress bar.
    - [x] Parallel version.
- [ ] Training neural network.
    - [x] Data generator. Read the pickle file and provide the data to the neural network in an appropriate form.
    - [x] Data augmentation in data generator.
    - [x] Allows users to import **custom models** written in python files via configuration file.
    - [ ] Allows users to import **custom callbacks** written in python files via configuration file.
    - [x] Save weights of neural network.
- [ ] Predict resistivity.
    - [x] Save data as pickle files.
    - [x] Progress bar.
    - [ ] Parallel version.
- [x] Predict resistance (potential difference divided by current)
    - [x] Save data as pickle files.
    - [x] Progress bar.
    - [x] Parallel version.
- [ ] Plot resistivity.
    - [x] synthetic data
        - [x] crossplot
        - [x] contour
        - [x] heatmap
    - [ ] filed data
        - [ ] crossplot
        - [ ] contour
    - [ ] Parallel version
    - [ ] Make figures more aesthetic.
- [x] Go to tensorflow 2.0
- [x] Change forward modeling code from FW2_5D to simpeg.
    - [x] Accept **flat** topography defined in terrain file
    - [ ] Accept topography defined in terrain file
- [ ] Build docker image. (optional)
- [ ] Publishing to PyPi. (optional)
- [ ] GUI (optional)