import numpy as np
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model
import dosma as dm
import os
import glob
from pathlib import Path
import time
from copy import deepcopy, copy
import pandas as pd

# Configuration and Paths
home_data_folder = "/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/DATA/cross_sectional_rand_50'"
home_save_path = "/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/DATA/cross_sectional_rand_50'"
list_participant_folders = glob.glob(os.path.join(home_data_folder, '9*'))

nifti_name = "t2_4d_array.nii"
mask_name = "registered_dess_segmentation.nii"
t2map_name = "t2.nii.gz"
b1map_name = "b1.nii.gz"

# Neural Network Configuration
home_nn_path = "/bmrNAS/people/barma7/Lab-work/Projects/OAI_T2mapping/repository_JMRI/code/deep_learning_training/results"
w_run = "SINC/TBW2/var_1e-9_1e-2/mae_loss"

sv_root = "epg_dl"
sv_fldr = copy(w_run)
sv_name = os.path.join(sv_root, sv_fldr)

# NN Parameters
weights_path = os.path.join(home_nn_path, w_run, 'weights', 'MODEL.500.hdf5')
Normalization = True
b1_estimation = True

# Sequence parameters
ETL = 7  # Echo Train Length
TE = 10  # Echo Time [ms]

# Define the Neural Network Model
def network_model():
    """Defines the architecture of the neural network model."""
    my_input = Input(shape=(Nb_pulses,), name='input')
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
    x = Dense(N_out, activation='linear', kernel_initializer='glorot_normal', name='output')(x)
    return my_input, x

# Initialize the Model
Nb_pulses = 7
Input_channels = 1  # 1 for magnitude, 2 for real + imaginary
N_out = 2 if b1_estimation else 1

input_stream, predictions = network_model()  # Create model architecture
CartilageNet = Model(inputs=input_stream, outputs=predictions)  # Create the model
CartilageNet.load_weights(weights_path)  # Load pre-trained weights

# Data Processing
list_sub_id, list_time, list_status, list_msg = [], [], [], []
strt_tot = time.time()

# Process Each Participant Folder
for fldr in list_participant_folders:
    sub_id = os.path.basename(fldr)
    print(f'Processing subject {sub_id}')
    
    if os.path.exists(os.path.join(fldr, nifti_name)):
        SvFldr = os.path.join(home_save_path, sub_id, sv_name)
        Path(SvFldr).mkdir(parents=True, exist_ok=True)

        # Load Data
        nr = dm.NiftiReader()
        data_nifti = nr.load(os.path.join(fldr, nifti_name))
        mask_nifti = nr.load(os.path.join(fldr, mask_name))

        # Preprocess Data
        Data = np.squeeze(deepcopy(data_nifti.volume))
        Mask = deepcopy(mask_nifti.volume)

        seg_list = np.unique(Mask)
        target_list = [1, 2, 3]

        if set(target_list).issubset(seg_list):
            nb_slices, nb_row, nb_col, nb_echoes = Data.shape[2], Data.shape[0], Data.shape[1], Data.shape[3]

            Data = np.reshape(Data, (nb_row * nb_col * nb_slices, nb_echoes))
            Mask = np.reshape(Mask, (nb_row * nb_col * nb_slices, 1))
            Idx = np.where((Mask > 0) & (Mask < 4))
            data = deepcopy(Data[Idx[0], :])

            # Normalize Data if Required
            if Normalization:
                data_temp = np.divide(data, np.linalg.norm(data, axis=1)[:, None])

            # Predict Quantitative Maps
            print('Computing quantitative maps')
            strt = time.time()
            prediction = CartilageNet.predict(data_temp)
            stp = time.time()

            print(f'Time for prediction: {stp - strt:.2f} seconds')
            list_time.append(stp - strt)

            pred = np.array(prediction)
            pred[pred < 0] = 0  # Crop negative values
            pred[:, 0] *= 10  # Scale T2 values

            # Apply Range Constraints
            pred[pred[:, 0] > 110, 0] = 110
            pred[pred[:, 0] < 10, 0] = 10

            # Initialize Maps
            T2 = np.zeros((nb_row * nb_col * nb_slices, 1))
            T2[Idx] = pred[:, 0]

            if b1_estimation:
                B1 = np.zeros((nb_row * nb_col * nb_slices, 1))
                B1[Idx] = pred[:, 1]

            # Reshape Maps
            T2map = np.reshape(T2, (nb_row, nb_col, nb_slices))
            if b1_estimation:
                B1map = np.reshape(B1, (nb_row, nb_col, nb_slices))

            # Save Quantitative Maps
            nwr = dm.NiftiWriter()
            T2map_mv = dm.MedicalVolume(T2map, data_nifti.affine)
            nwr.save(T2map_mv, os.path.join(SvFldr, t2map_name))

            if b1_estimation:
                B1map_mv = dm.MedicalVolume(B1map, data_nifti.affine)
                nwr.save(B1map_mv, os.path.join(SvFldr, b1map_name))

            # Save Fitted Parameters
            np.savetxt(os.path.join(SvFldr, 'T2fit.txt'), pred[:, 0])
            if b1_estimation:
                np.savetxt(os.path.join(SvFldr, 'B1fit.txt'), pred[:, 1])

            msg = 'Subject processed successfully'
            status = 2
        else:
            msg = 'Subject does not have all the required labels, skipping subject'
            print(msg)
            status = 1
            list_time.append(0)
    else:
        msg = 'File does not exist, skipping subject'
        print(msg)
        status = 0
        list_time.append(0)
    
    # Log the processing status for each subject
    list_sub_id.append(sub_id)
    list_status.append(status)
    list_msg.append(msg)

# Create a DataFrame to log the processing status
df = pd.DataFrame(list(zip(list_sub_id, list_status, list_msg, list_time)), 
                  columns=['sub_id', 'status', 'processing time', 'message'])
df.to_csv(os.path.join(home_save_path, 'processing_log_EPG_DL.csv'), index=False)

end_tot = time.time()
print(f"Time to process all subjects: {end_tot - strt_tot:.2f} seconds")
