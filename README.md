# ERINN
Electrical resistivity imaging based on deep learning.

ERINN contains convenient functions for using deep learning to solve ERI problems.
The result can then provide a quick view of the resistivity and an useable initial model in the inversion routine.

Here are the specific steps:
1. Generate data using the matlab program 'FW2_5D'.
2. Preprocess the data and split it into training/testing.
3. Train the neural network.
4. Verify the quality of the prediction with some figures and metrics.

## Requirement
- matlab >= 2016b
- python >= 3.6
- matplotlib >= 3.0.3
- numpy >= 1.16.2
- numba >= 0.43.0
- ruamel.yaml >= 0.16.5
- tensorflow = 2.0.0

## ERI project (matlab verison)
Under the ERI directory, there is a template named {template_matlab}.  
You can use the template directly and  change the name of the template to a meaningful name, 
such as a field name or an experiment name.
    
This template has some useful scripts:
- [generate_required_data.m](ERI/template_matlab/scripts/generate_required_data.m)
  - Generate synthetic resistivity data and 
    equivalent resistance(potential difference divided by current) receive by electrode array.
  - Generate essential global parameters.
- [generate_processed_data.py](ERI/template_matlab/scripts/generate_processed_data.py)
  - Generate input and output data for neural network.
- [pre_training.py](ERI/template_matlab/scripts/pre_training.py)
  - Pre-train the neural network using small dataset.
- [training.py](ERI/template_matlab/scripts/training.py)
  - Train the neural network with the pre-trained weight.
- [predict_resistivity.py](ERI/template_matlab/scripts/predict_resistivity.py)
  - Predict the underground resistivity.
- [predict_potential.m](ERI/template_matlab/scripts/predict_potential.m)
  - Use predictive resistivity to simulate potential and get equivalent resistance.
- [plot_subsurface.m](ERI/template_matlab/scripts/plot_subsurface.m)
  - Plot some figures to check the robustness of neural network.

---
The following are definitions of the parameters in [config.json](ERI/template_matlab/config/config.json):
- output_path:   
  The directory that stores synthetic data.

- nx:  
  The number of grids in the x direction in the forward model.

- nz:  
  The number of grids in the z direction in the forward model.

- geometry_urf:  
  Electrode array geometry is extracted from this urf file.

- array_type:  
  Electrode configuration for collecting data. 
  
- Para_mat:  
  The mat file used to save the fw2_5D parameter.

- core:  
  A rule that produces resistivity. 

- samples:  
  Number of resistivity models and corresponding data.

- lower_bound:  
  The lower bound of uniform distribution at the log10 scale.

- upper_bound:  
  The upper bound of uniform distribution at the log10 scale.

- block_x_min:  
  Minimum length of block in the x direction.

- block_x_max:  
  Maximum length of block in the x direction.

- block_z_min:  
  Minimum length of block in the z direction.

- block_z_max:  
  Maximum length of block in the x direction.
---
Since the package is under construction, many files are old versions.  
Please star us for upcoming functionality.

---
## Note that we will discard the matlab part and rewrite the entire package in python.
### Scheduled milestone

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
- [ ] Predict resistance (potential/current)
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
- [ ] Build docker image. (optional)
- [ ] Publishing to PyPi. (optional)
- [ ] GUI (optional)