#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for data handling and preprocessing.
"""
import numpy as np
from os.path import join
import scipy.io  
import random
from copy import deepcopy
from config import (data_path, B, s, r, exp_folder, Normalization, Nb_pulses, 
                    AddNoise, var_min, var_max, training_size, shuffle, 
                    Input_channels)

# Load ground truth (GT) values
# The GT file contains (T1, T2) values for each dictionary entry.
# These values are used as the labels during training.
GT_ = np.float32(np.loadtxt(join(data_path, exp_folder, 'LUT.txt'), delimiter=','))
GT = deepcopy(GT_)

# Scale T2 values by dividing by 10 to make them comparable to B1 values.
# This scaling is done to bring the T2 values closer to the scale of B1 values.
GT[:, 0] /= 10

# Split data into training and test sets
n = GT.shape[0]  # Number of samples in GT
indices = np.arange(n)  # Create an array of indices for the samples

# If shuffle is enabled, shuffle the indices to randomize the data split
if shuffle:
    random.seed(s)  # Set the random seed for reproducibility
    random.shuffle(indices)

# Split the data into training and test sets based on the `training_size` parameter.
# If `training_size` is specified, use that value to determine the split.
# Otherwise, use the ratio `r` to determine the split.
if training_size:
    x_train = indices[:training_size]
    x_test = indices[training_size:]
else:
    split_index = int(r * n)  # Determine the index for splitting the data
    x_train = indices[:split_index]  # Training indices
    x_test = indices[split_index:]  # Test indices

# Load and preprocess training data
# The data is loaded from a MATLAB file and is expected to be complex-valued.
Data_train = scipy.io.loadmat(join(data_path, exp_folder, 'dictionary.mat'))['dictionary'][:, :Nb_pulses]
Data_train = np.complex64(Data_train)  # Ensure data is in complex64 format

# Normalize the data so that each signal has a norm of 1.
Data_train /= np.linalg.norm(Data_train, axis=1)[:, None]

def data_generator(train=True):
    """
    Generator that yields batches of data for Keras.
    This function is used to feed data into the model during training.
    It continuously generates data batches, which is useful for large datasets.
    """
    while True:
        inputs, labels = get_data(train=train)  # Fetch a batch of data
        yield inputs, labels  # Yield the data to the Keras model

def get_data(train=True):
    """
    Fetch a batch of data and optionally apply data augmentation.
    This function selects a batch of samples from the training or test set,
    applies any specified data augmentation, and returns the processed data and labels.
    """
    x_ = x_train if train else x_test  # Select training or test indices
    sampled_indices = random.sample(list(x_), np.minimum(B, len(x_)))  # Randomly sample a batch of indices

    # Extract the data and labels for the sampled indices
    data_tensor = Data_train[sampled_indices]
    labels_tensor = GT[sampled_indices]

    # If noise addition is enabled, add Gaussian noise to the data
    if AddNoise:
        sampled_var = np.random.uniform(var_min, var_max, len(sampled_indices))
        data_tensor = add_gaussian_noise_to_signal_var(data_tensor, sampled_var)

    # Process the data based on the number of input channels (magnitude or real+imaginary)
    if Input_channels == 2:
        if Normalization:  # Normalize the data if required
            data_tensor /= np.linalg.norm(data_tensor, axis=1)[:, None]
        # Separate the real and imaginary parts
        real_data_tensor = np.real(data_tensor)
        imag_data_tensor = np.imag(data_tensor)
        # Concatenate real and imaginary parts to form the final input tensor
        data_tensor = np.concatenate((real_data_tensor, imag_data_tensor), axis=1)
    else:
        if Normalization:  # Normalize the magnitude of the data if required
            data_tensor = np.abs(data_tensor)
            data_tensor /= np.linalg.norm(data_tensor, axis=1)[:, None]

    return data_tensor, labels_tensor  # Return the processed data and labels

def add_gaussian_noise_to_signal_var(signal, var):
    """
    Add Gaussian noise to a complex signal with specified variance.
    This function adds noise separately to the real and imaginary parts of the signal.
    """
    if np.iscomplexobj(signal):
        # Calculate the noise for the real and imaginary parts separately
        noise_real = np.random.randn(signal.shape[0], signal.shape[1]) * np.sqrt(var / 2)[:, None]
        noise_imag = np.random.randn(signal.shape[0], signal.shape[1]) * np.sqrt(var / 2)[:, None]
        # Add the noise to the signal
        noisy_signal = np.complex64(signal + (noise_real + 1j * noise_imag))
    else:
        # If the signal is real, add noise directly to it
        noise_real = np.random.randn(signal.shape[0], signal.shape[1]) * np.sqrt(var)[:, None]
        noisy_signal = np.float32(signal + noise_real)

    return noisy_signal  # Return the noisy signal
