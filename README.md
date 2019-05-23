# ERINN
Electrical resistivity imaging based on deep learning.


## Requirement
- matlab >= 2016b
- python >= 3.6
- tensorflow = 1.13.1
- numpy >= 1.16.2
- numba >= 0.43.0

## ERI project
Under the ERI directory, there is a template named {project}.  
You can use the template directly and  change the name of the template to a meaningful name, 
such as a field name or an experiment name.
    
This template has some useful scripts.
- [generate_required_data.m](ERI/{project}/scripts/generate_required_data.m)
  - Generate synthetic resistivity data and 
    equivalent resistance(potential difference divided by current) receive by electrode array.
  - Generate essential global parameters.
- [generate_processed_data.py](ERI/{project}/scripts/generate_processed_data.py)
  - Generate input and output data for neural network.
- [pre_training.py](ERI/{project}/scripts/pre_training.py)
  - Pre-train the neural network using small dataset.
- [training.py](ERI/{project}/scripts/training.py)
  - Train the neural network with the pre-trained weight.
- [predict_resistivity.py](ERI/{project}/scripts/predict_resistivity.py)
  - Predict the underground resistivity.
- [predict_potential.m](ERI/{project}/scripts/predict_potential.m)
  - Use predictive resistivity to simulate potential and get equivalent resistance.
- [plot_subsurface.m](ERI/{project}/scripts/plot_subsurface.m)
  - Plot some figures to check the robustness of neural network.


 
Since the package is under construction, many files are old versions.
Please star us for upcoming functionality.

