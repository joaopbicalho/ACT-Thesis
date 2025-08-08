# Adaptive Chirplet Transform for EEG Sleep Stage Classification

**Undergraduate Thesis **  
**Supervised by:** Prof. Steve Mann  

## Overview

This repository contains a comprehensive implementation of the **Adaptive Chirplet Transform (ACT)** for EEG signal analysis and automatic sleep stage classification. The project explores the application of time-frequency-chirp rate analysis to extract meaningful features from EEG signals for sleep staging tasks.

The implementation includes both CPU and GPU-accelerated versions of the ACT algorithm, along with deep learning models for sleep stage classification.

## What is the Adaptive Chirplet Transform?

The Adaptive Chirplet Transform is an advanced signal processing technique that decomposes signals into time-frequency-chirp rate atoms called "chirplets." Unlike traditional time-frequency representations, ACT captures both the instantaneous frequency and the rate of frequency change (chirp rate) of signal components, making it particularly suitable for analyzing non-stationary signals like EEG.

**Key advantages:**
- Superior time-frequency-chirp rate resolution
- Adaptive basis functions that match signal characteristics
- Sparse representation of complex signals
- Robust feature extraction for machine learning applications

## Project Structure

```
📦 ACT-Thesis/
├── 📁 CPU-Implementation/          # CPU-based ACT implementation
│   ├── act.py                      # Core ACT library and algorithms [3]
│   └── act_processing.py           # EEG processing pipeline (CPU) [3]
├── 📁 GPU-Implementation/          # GPU-accelerated implementation
│   ├── act.py                      # GPU-optimized ACT using CuPy
│   └── act_processing_gpu.py       # EEG processing pipeline (GPU)
├── 📁 ANN/                         # Neural network models
│   ├── nn4.py                      # Deep learning model for classification
│   └── split_for_training.py       # Data preprocessing and splitting
├── 📁 utils/                       # Utility scripts and tools
│   ├── data_label.py              # Label matching and annotation
│   ├── input_transform.py         # Input transformation layers
│   ├── join_labels_fft.py         # FFT feature integration (not used for ACT)
│   └── run_all.sh                 # Batch processing script
└── README.md                       # This file
```

## Implementation Details

### Core ACT Algorithm (`CPU-Implementation/` & `GPU-Implementation/`)

#### **act.py** - Core ACT Library
- **Chirplet Generation**: Creates adaptive basis functions with time, frequency, and chirp rate parameters
- **Dictionary Generation**: Builds comprehensive chirplet dictionaries for signal decomposition
- **Matching Pursuit**: Implements greedy algorithms for optimal chirplet selection
- **Transform Functions**: Performs P-order ACT approximations of input signals

**Key Parameters:**
- `FS`: Sampling frequency (default: 256 Hz)
- `length`: Signal length in samples (default: 3840 for 15-second epochs)
- `tc_info`: Time center range (min, max, step)
- `fc_info`: Frequency center range (0.5-20 Hz for EEG)
- `logDt_info`: Log duration range (-4 to -1)
- `c_info`: Chirp rate range (-30 to 30)

#### **act_processing.py / act_processing_gpu.py** - EEG Processing Pipeline
- **Data Loading**: Reads EDF files using MNE-Python
- **Preprocessing**: Applies bandpass filtering (1-40 Hz) and notch filtering (60 Hz)
- **Epoching**: Segments data into 15-second epochs with 75% overlap
- **ACT Feature Extraction**: Applies ACT to each epoch and extracts coefficients and parameters
- **Batch Processing**: Handles multiple subjects (1-128) from the BitBrain dataset

### Neural Network Models (`ANN/`)

#### **nn4.py** - Deep Learning Classification Model
- **Custom Dataset Class**: `EEGSequenceDataset` handles ACT-processed features
- **Sequence Modeling**: Supports temporal sequence learning for improved classification
- **Class Balancing**: Implements class weighting for imbalanced sleep stage data
- **Multi-class Classification**: Classifies into standard sleep stages (Wake, N1, N2, N3, REM)

#### **split_for_training.py** - Data Preprocessing
- **Feature Normalization**: Standardizes ACT coefficients and parameters
- **Data Splitting**: Creates train/validation/test splits with sequence awareness
- **Label Encoding**: Maps sleep stage annotations to numerical labels

### Utility Functions (`utils/`)

#### **data_label.py** - Label Integration
- **Epoch-Label Matching**: Aligns 15-second epochs with 30-second sleep stage annotations
- **Cross-boundary Handling**: Discards epochs that span multiple sleep stages that disagree
- **Temporal Validation**: Ensures consistent labeling across time series

#### **input_transform.py** - Model Integration
- **Feature Transformation**: Converts ACT features for compatibility with existing models, not used for the final NN model and final results.

## Installation and Dependencies

### Required Packages

```bash
# Core scientific computing
numpy>=1.21.0
scipy>=1.7.0
pandas>=1.3.0
matplotlib>=3.4.0

# EEG signal processing
mne>=0.23.0

# Deep learning
torch>=1.9.0
scikit-learn>=0.24.0

# GPU acceleration (optional)
cupy-cuda11x>=9.0.0  # For CUDA 11.x

# Utilities
tqdm>=4.62.0
joblib>=1.0.0
```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/joaopbicalho/ACT-Thesis.git
cd ACT-Thesis
```

2. **Install dependencies:**
```bash
pip install numpy scipy pandas matplotlib mne torch scikit-learn tqdm joblib
```

3. **For GPU acceleration (optional):**
```bash
pip install cupy-cuda11x  # Adjust CUDA version as needed
```

## Usage

### Basic EEG Processing

#### CPU Implementation
```bash
cd CPU-Implementation
python act_processing.py
```

#### GPU Implementation
```bash
cd GPU-Implementation
python act_processing_gpu.py [subject_id]
```

### Batch Processing
```bash
cd utils
chmod +x run_all.sh
./run_all.sh  # Processes all 128 subjects
```

### Neural Network Training
```bash
cd ANN
python split_for_training.py  # Prepare data splits
python nn.py                 # Train classification model
```

## Research Contributions

1. **Novel Application**: First comprehensive application of ACT to sleep EEG analysis
2. **GPU Implementation**: High-performance computing solution for real-time processing
3. **Feature Engineering**: ACT-based features capture sleep-specific signal characteristics

## References

1. Mann, S., & Haykin, S. (1995). The chirplet transform: Physical considerations. *IEEE Transactions on Signal Processing*, 43(11), 2745-2761.
2. BitBrain Sleep Dataset: https://github.com/OpenNeuroDatasets/ds005555
3. Bhargava and S. Mann, “Adaptive chirplet transform-based machine learning for p300
brainwave classification,” in 2020 IEEE-EMBS Conference on Biomedical Engineering and
Sciences (IECBES), pp. 62–67, 2021 



