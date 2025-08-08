import act as act_lib
import numpy as np
import pandas as pd
import mne
import csv
import matplotlib.pyplot as plt


# Initialize ACT with appropriate parameters
act = act_lib.ACT(
    FS=256,
    length=512,    # 3840 samples for 15 seconds with a sampling frequency of 256 Hz
    tc_info=(0, 512, 20),  # Reduced range and step size
    fc_info=(0.5, 20, 0.1),  # Adjusted step size
    logDt_info=(-4, -1, 0.3),  # Adjusted step size
    # c_info=(-30, 30, 1),  # Adjusted step size
    force_regenerate=True,
    mute=False
)


# Load the data from the CSV file
data = pd.read_csv('joao_testing_0814.csv')

data.replace('', np.nan, inplace=True)
data.ffill(inplace=True)
data.bfill(inplace=True)

eeg_columns = ['eeg_1', 'eeg_2', 'eeg_3', 'eeg_4']
eeg_data = data[eeg_columns].values 

# Create an MNE RawArray
info = mne.create_info(ch_names=eeg_columns, sfreq=256, ch_types='eeg')
raw = mne.io.RawArray(eeg_data.T, info)

# Function to apply FIR bandpass filter using MNE
raw.filter(l_freq=1.0, h_freq=40.0, fir_design='firwin', fir_window='hamming', phase='zero', method='fir')
# Notch filter to remove power line noise 
raw.notch_filter(freqs=60, fir_design='firwin')

# Extract data and times
eeg_data, times = raw[:, :]  

# eeg_data is  a NumPy array with shape (n_samples, n_channels)
eeg_data = eeg_data.T
# Determine the number of epochs
num_samples = eeg_data.shape[0]
epoch_length = 512
num_epochs = num_samples // epoch_length

# Open a CSV file to write the results
with open('act_results_20fc.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    
    # Write the header row
    header = ['Epoch', 'Params', 'Coeffs']
    writer.writerow(header)

    # Process each epoch
    for epoch_idx in range(num_epochs):
        start_idx = epoch_idx * epoch_length
        end_idx = start_idx + epoch_length
        
        print(f"Processing epoch {epoch_idx + 1}/{num_epochs}")
        
        for electrode_idx, electrode_data in enumerate(eeg_columns):
            segment = eeg_data[start_idx:end_idx, electrode_idx]
            
            result = act.transform(segment, order=5, debug=False)

            # Plot original signal and approximation
            plt.figure(figsize=(10, 5))
            plt.plot(segment, label='Original Signal')
            plt.plot(result['approx'], label='Approximation', linestyle='--')
            plt.title(f'Epoch {epoch_idx + 1}, Electrode {electrode_data}')
            plt.xlabel('Samples')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.show()


            #  # Prepare the result data for CSV
            # params = result['params'].flatten().tolist()
            # coeffs = result['coeffs'].flatten().tolist()
            # signal = result['signal'].flatten().tolist()
            # error = result['error']
            # residue = result['residue'].flatten().tolist()
            # # approx = result['approx'].flatten().tolist()
            
            # # Combine all result data into a single row
            # row = [params, coeffs, signal, error, residue]
            
            # # Write the row to the CSV file
            # writer.writerow(row)
            
            # # Print or store the result as needed
            print(f"Electrode {electrode_idx + 1} result: {result}")
 
