# Import statements
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
from keras.callbacks import CSVLogger, ModelCheckpoint
from pathlib import Path
from os.path import join
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Import functions and parameters from utility and configuration files
from utils import data_generator
from config import (N, N_out, lr_init, dev_set, homeDir, save_path, 
                    data_path, loss, optimizer, B, s, r, exp_folder, 
                    shuffle, Normalization, AddNoise, var_min, var_max, 
                    Nb_pulses, Input_channels, training_size)

# Define the network architecture
def network_model():
    """
    This function defines the architecture of the neural network model.
    The model consists of multiple Dense (fully connected) layers with LeakyReLU activation.
    """
    # Input layer
    my_input = Input(shape=(N,), name='input')
    
    # Hidden layers with LeakyReLU activations
    x = Dense(512, kernel_initializer='glorot_normal', name='hidden0')(my_input)
    x = LeakyReLU(name='hidden0_1')(x)
    x = Dense(256, kernel_initializer='glorot_normal', name='hidden1')(x)
    x = LeakyReLU(name='hidden1_1')(x)
    x = Dense(128, kernel_initializer='glorot_normal', name='hidden2')(x)
    x = LeakyReLU(name='hidden2_1')(x)
    x = Dense(64, kernel_initializer='glorot_normal', name='hidden3')(x)
    x = LeakyReLU(name='hidden3_1')(x)
    x = Dense(32, kernel_initializer='glorot_normal', name='hidden4')(x)
    x = LeakyReLU(name='hidden4_1')(x)
    
    # Output layer with linear activation
    x = Dense(N_out, activation='linear', kernel_initializer='glorot_normal', name='output')(x)
    
    # Return the input and output tensors
    return my_input, x

# Instantiate the model
input_stream, predictions = network_model()  # Create the tensors by calling the network_model5 function
model = Model(inputs=input_stream, outputs=predictions)  # Create the model to be trained by Keras
model.summary()  # Print a summary of the model architecture

# Compile the model
# Create directories to save logs and weights if they don't exist
Path(join(save_path, "log_loss")).mkdir(parents=True, exist_ok=True)
Path(join(save_path, "weights")).mkdir(parents=True, exist_ok=True)

# Define the optimizer (Adam) with specified learning rate and other parameters
adam_opt = Adam(lr=lr_init, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Compile the model with the specified loss function and optimizer, and track mean absolute percentage error (MAPE)
model.compile(loss=loss, optimizer=adam_opt, metrics=['mape'])

# Define a checkpoint callback to save the model weights periodically
checkpoint = ModelCheckpoint(join(save_path, "weights", "MODEL.{epoch:02d}.hdf5"), save_weights_only=True, period=250)

# Set up a CSV logger to record the training and validation loss for each epoch
csv_logger = CSVLogger(join(save_path, "log_loss", "log_loss_MODEL.csv"), separator=',', append=False)

# Optionally load weights from a previous checkpoint (commented out by default)
# weightsPath = join(homeDir,"Deep_Learning/MLP/training_results/v2/MODEL6/50000_10_110/var_1e-9_1e-2/EPG/weights")
# model.load_weights(join(weightsPath, 'MODEL.500.hdf5'))

# Save the configuration parameters to a CSV file for reference and reproducibility
params = {
    'homeDir': homeDir, 'save_path': save_path, 'data_path': data_path, 'exp_folder': exp_folder,
    'shuffle': shuffle, 'Normalization': Normalization, 'AddNoise': AddNoise, 'var_min': var_min,
    'var_max': var_max, 'Nb_pulses': Nb_pulses, 'Input_channels': Input_channels, 'N': N, 'N_out': N_out,
    'dev_set': dev_set, 'training_size': training_size, 'seed': s, 'ratio': r, 'BatchSize': B,
    'lr_init': lr_init, 'optimizer': optimizer, 'loss': loss
    }

# Convert the parameters dictionary to a DataFrame and save it as a CSV file
df = pd.DataFrame.from_dict(params, orient='index')
df.to_csv(join(save_path, 'params.csv'), header=False)

# Train the model
if dev_set:
    # If dev_set is True, use a validation set during training
    model.fit(
        data_generator(train=True), 
        steps_per_epoch=1000, 
        epochs=500, 
        verbose=1, 
        callbacks=[csv_logger, checkpoint],
        validation_data=data_generator(train=False), 
        validation_steps=500, 
        initial_epoch=0
    )
else:
    # If dev_set is False, train without a validation set
    model.fit(
        data_generator(train=True), 
        steps_per_epoch=1000, 
        epochs=500, 
        verbose=1, 
        callbacks=[csv_logger, checkpoint], 
        initial_epoch=0
    )
