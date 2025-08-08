import sys
import act as act_lib
import numpy as np
import cupy as cp
import pandas as pd
import mne
import csv
import os
import time
import matplotlib.pyplot as plt

# Set MNE log level to ERROR to minimize verbosity
mne.set_log_level("ERROR")

# Read subject id from command-line arguments
if len(sys.argv) > 1:
    subject_id = sys.argv[1]
else:
    subject_id = "1"  # default to subject 1 if no parameter is provided

# Start timing
start_time = time.time()

epoch = 15  # Epoch length in seconds
overlap = 0.75  # 75% overlap

# Create a GPU-enabled ACT object
act = act_lib.ACT(
    FS=256,
    length=epoch * 256,
    tc_info=(0, epoch * 256, 64),
    fc_info=(0.6, 15, 0.2),
    logDt_info=(-4, 0, 0.3),
    c_info=(-10, 10, 0.25),
    force_regenerate=True,
    mute=False,
)

# Build the file path from the subject id
current_dir = os.getcwd()
data_file = f"sub-{subject_id}/eeg/sub-{subject_id}_task-Sleep_acq-headband_eeg.edf"
file_dir = os.path.join(current_dir, data_file)
print("Processing file:", file_dir)

# Load the EDF data (with verbose disabled)
raw_data = mne.io.read_raw_edf(file_dir, preload=True, verbose=False)

# Only 2 channels in this data
eeg_channels = ["HB_1", "HB_2"]
raw_data.pick_channels(eeg_channels)

# Apply bandpass and notch filters with verbosity disabled
raw_data.filter(
    l_freq=1.0,
    h_freq=40.0,
    fir_design="firwin",
    fir_window="hamming",
    phase="zero",
    method="fir",
    l_trans_bandwidth=0.5,
    h_trans_bandwidth=12.5,
    verbose=False
)
raw_data.notch_filter(freqs=50, fir_design="firwin", verbose=False)

# Extract EEG data and transpose so rows are samples, columns are channels
eeg_data, times = raw_data[:, :]
eeg_data = eeg_data.T  # shape: (samples, channels)

# Convert EEG data to a GPU (CuPy) array for processing
eeg_data_gpu = cp.asarray(eeg_data, dtype=cp.float32)

# Determine epoch parameters
epoch_length = epoch * act.FS  # e.g., 15 seconds * 256 Hz = 3840 samples
stride = int(epoch_length * (1 - overlap))  # step size based on overlap

# Create a Hamming window as a NumPy array then convert to a CuPy array
hamming_window = cp.asarray(np.hamming(epoch_length), dtype=cp.float32)

# Open a CSV file to write the results (results are written from CPU)
output_csv = f"act_results_sub-{subject_id}_attempt.csv"
with open(output_csv, mode="w", newline="") as file:
    writer = csv.writer(file)
    header = ["Epoch", "Params", "Coeffs", "Signal", "Error", "Residue"]
    writer.writerow(header)

    # Process each overlapping epoch
    for start_idx in range(0, int(eeg_data.shape[0]) - epoch_length + 1, stride):
        end_idx = start_idx + epoch_length
        epoch_idx = start_idx // stride + 1
        print(f"Processing epoch {epoch_idx}")

        # Process each electrode
        for electrode_idx, electrode_name in enumerate(eeg_channels):
            # Extract the epoch segment for the current electrode from the GPU array
            segment_gpu = eeg_data_gpu[start_idx:end_idx, electrode_idx]
            # Apply the Hamming window on GPU
            windowed_segment_gpu = segment_gpu * hamming_window

            # Apply ACT transform (expects a CuPy array)
            result = act.transform(windowed_segment_gpu, order=6, debug=True)

            # Convert results from GPU to CPU (NumPy) for CSV output and plotting
            params = cp.asnumpy(result["params"]).flatten().tolist()
            coeffs = cp.asnumpy(result["coeffs"]).flatten().tolist()
            signal = cp.asnumpy(result["signal"]).flatten()
            error = result["error"]  # assumed to be a scalar (CPU float)
            residue = cp.asnumpy(result["residue"]).flatten().tolist()

            row = [epoch_idx, params, coeffs, signal.tolist(), error, residue]
            writer.writerow(row)

            loop_time = time.time() - start_time
            print(f"Loop time: {loop_time:.2f} seconds")

end_time = time.time()
print(f"Total processing time: {end_time - start_time:.2f} seconds")
