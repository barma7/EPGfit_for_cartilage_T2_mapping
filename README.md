# EPGfit_for_cartilage_T2_mapping
This repository contains the code and resources for performing cartilage T2 mapping from simulated Multi-Echo Spin Echo (MESE) data using the Extended Phase Graph (EPG), as described in Marco Barbieri, Anthony A. Gatti, and Feliks Kogan's work "Improving Accuracy and Repeatability of Cartilage T2 Mapping in the OAI Dataset through Extended Phase Graph Modeling." 

The article is currently in press in the Journal of Magnetic Resonance Imaging, but the full citation isn't still available and will be updated soon. If you use code from this repository, please cite the journal publication.

Full citation: **Marco Barbieri, Anthony A. Gatti and Feliks Kogan. Improving Accuracy and Repeatability of Cartilage T2 Mapping in the OAI Dataset through Extended Phase Graph Modeling.  J. Magn. Reson. Imaging (2024). In press.**

The pre-computed T2 maps using the EPG-dictionary method are being publicly released in Hugging Face at: https://huggingface.co/datasets/barma7/oai-t2maps-epgfit

## EPG Formalism for Cartilage T2 Mapping from MESE Data
The repository includes code for training a Multi-Layer Perceptron (MLP) model and performing EPG fit using dictionary matching and non-linear least squares methods.
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

1. Barbieri et al. Improving accuracy and reproducibility of cartilage T2 mapping in the OAI dataset through extended phase graph modeling. Journal of Magnetic Resonance Imaging, 2024, in Press.
2. StimFit toolbox: https://github.com/usc-mrel/StimFit
