# EPGfit_for_cartilage_T2_mapping

## EPG Formalism for Cartilage T2 Mapping from MESE Data

This repository contains the code and resources for performing cartilage T2 mapping from simulated Multi-Echo Spin Echo (MESE) data using the Extended Phase Graph (EPG) formalism. The repository includes code for training a Multi-Layer Perceptron (MLP) model as well as for performing EPG fit using dictionary matching and non-linear least squares methods.

### Directory Structure

1. **deep_learning_training**: This folder contains the Python code for training the MLP model for cartilage T2 mapping from simulated MESE data using the EPG formalism.

2. **epg_utils**: This folder contains MATLAB utilities used to perform EPG simulations. The core EPG function is taken from the StimFit toolbox (https://github.com/usc-mrel/StimFit). The simulations are provided for SINC pulses with Time-Bandwidth (TBW) product of 2.

3. **sim-data**: This folder contains dictionaries used to perform dictionary matching and to train the NN model.

4. **epg_fit**: This folder contains the code to perform EPG fit as described in the paper "Improving accuracy and reproducibility of cartilage T2 mapping in the OAI dataset through extended phase graph modeling" using three different methods: deep learning, dictionary matching, and non-linear least squares. The data can be downloaded from Zenodo at https://doi.org/10.5281/zenodo.13351169.

### Usage

1. **deep_learning_training**: To train the MLP model, run the Python script in this folder.

2. **epg_utils**: The MATLAB utilities in this folder can be used to perform EPG simulations for SINC pulses with TBW = 2.

3. **sim-data**: The dictionaries in this folder can be used for dictionary matching and training the NN model.

4. **epg_fit**: The code in this folder can be used to perform EPG fit using deep learning, dictionary matching, and non-linear least squares methods. The data required for this can be downloaded from Zenodo.

### Dependencies

- Python 3.x
- NumPy
- SciPy
- Keras
- TensorFlow
- MATLAB (for the `epg_utils` folder)

### References

1. Improving accuracy and reproducibility of cartilage T2 mapping in the OAI dataset through extended phase graph modeling. Journal of Magnetic Resonance Imaging, 2022.
2. StimFit toolbox: https://github.com/usc-mrel/StimFit
