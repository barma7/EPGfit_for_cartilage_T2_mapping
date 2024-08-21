#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for neural network training.
"""
from os.path import join

# Directories and file paths
homeDir = r"/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/code"
data_path = join(homeDir, "sim-data")
exp_folder = join(r"dictionaries/SINC/TBW2/SLR/rand/50000/1")

homeSv = r"/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/code/deep_learning_training/results"
save_path = join(homeSv, r"SINC/TBW2/var_1e-9_1e-2/mae_loss")

# Data handling settings
shuffle = False  # Shuffle the dataset
Normalization = True  # Normalize each signal to have a norm of one

# Data augmentation settings
AddNoise = True  # Add noise to the data
var_min = 1e-9  # Minimum variance for noise
var_max = 1e-2  # Maximum variance for noise

# Input and output dimensions
Nb_pulses = 7  # Number of MESE pulses
Input_channels = 1  # 1: magnitude, 2: real + imaginary

# Set input dimension based on number of channels
N = Nb_pulses if Input_channels == 1 else 2 * Nb_pulses
N_out = 2  # Output dimensions (e.g., T2, B1)

# Training parameters
dev_set = True  # Use test examples to estimate validation loss during training
training_size = False  # Set to False to use ratio r for training/test split
s = 0.12345  # Random seed for reproducibility
r = 8.5 / 10.0  # Training/testing data ratio
B = 500  # Batch size
lr_init = 0.0001  # Initial learning rate
optimizer = 'adam'  # Optimizer
loss = 'mae'  # Loss function
