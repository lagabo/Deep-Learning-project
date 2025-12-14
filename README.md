# Deep Learning Class (VITMMA19) Project Work - Bull Flag Detector

## Project Details

### Project Information

- **Selected Topic**: Bull-flag detector
- **Student Name**: László Gábor
- **Aiming for +1 Mark**: No

### Solution Description

This solution implements a deep learning system to recognize "bull flag" and "bear flag" patterns in financial time series data. There are six different type of patterns that the project used for learning. These categories are:
- Bull Flag: Normal, Wedge, Pennant
- Bear Flag: Normal, Wedge, Pennant

The dataset used for the project contains two different data types. One data is the 5 csv files that contain the time series data for the EUR/HUF currency pair. The 5 files contain different time ranges, and are named accordingly. The 5 time ranges are 1 minute, 5 minutes, 15 minutes, 30 minutes, and 1 hour. The other data is the json file, that contains the labels for the previously mentioned patterns, in the excel files.

### Data Preparation
The used data is preprocessed in the following way:
- Time series data is loaded from CSV files
- Label data is loaded from JSON file
- Price data is normalized to [0, 1] range using min-max normalization
- Sequences are padded to uniform length
- Data is split into train (70%), validation (15%), and test (15%) sets

### Model description
The following models were used for the project:
1. **Baseline Model (LSTM):**
   - 1-layer LSTM (batch_first=True), hidden size = 2
   - The last timestep output is used for classification
   - Dropout (p=0.3)
   - Fully connected output layer for 6-class classification

2. **Advanced Model (1D CNN):**
   - 2x Conv1D blocks with BatchNorm + ReLU
   - Conv1: in_channels=1, out_channels=16, kernel_size=3
   - Conv2: in_channels=16, out_channels=32, kernel_size=3
   - Global Average Pooling (AdaptiveAvgPool1d(1))
   - Dropout (p=0.3)
   - Fully connected output layer for 6-class classification

The models were booth trained with the following methodology:
- Optimizer: Adam with learning rate 0.001
- Loss function: Cross-Entropy Loss
- Batch size: 32
- Maximum epochs: 100
- Best model checkpoint is saved based on the validation loss
- Models are trained on the same train/validation split for fair comparison

The models were evaluated with the following metrics:
- Accuracy
- F1-Score (macro)
- Confusion Matrix
- Comparative comparison between LSTM and CNN models


### Docker Instructions

This project is containerized using Docker. Follow the instructions below to build and run the solution.

#### Build

Run the following command in the root directory of the repository to build the Docker image:

```bash
docker build -t dl-project .
```

#### Run

To run the solution, the container will automatically download the data from SharePoint and execute the full pipeline.

**To capture the logs for submission (required), redirect the output to a file:**

```bash
docker run --rm --mount type=bind,source="$(pwd)/data",target=/app/data dl-project > log/run.log 2>&1
```

### File Structure and Functions

The repository is structured as follows:

- **`src/`**: Contains the source code for the machine learning pipeline.
    - `01-data-preprocessing.py`: Donwload data from SharePoint, extracting, loading CSV/JSON files, normalization, padding, and train/val/test split.
    - `02-training.py`: Training loop for both LSTM baseline and 1D CNN models with model checkpointing.
    - `03-evaluation.py`: Comprehensive evaluation on test set with confusion matrix and comparative analysis.
    - `04-inference.py`: Inference on sample data with probability distributions for both models.
    - `models.py`: Model definitions for LSTM and 1D CNN classifiers.
    - `config.py`: Configuration file containing hyperparameters, and paths.
    - `utils.py`: Logger setup and helper functions (downloading data from SharePoint and loading preprocessed data).

- **`notebook/`**: Contains Jupyter notebooks for analysis and experimentation.
    - `01-data-exploration.ipynb`: Notebook for initial exploratory data analysis (EDA) and visualization.
    - `02-label-analysis.ipynb`: Notebook for analyzing the distribution and properties of the target labels.

- **`log/`**: Contains log files.
    - `run.log`: Log file showing the complete pipeline execution output.

- **`data/`**: Created at runtime, contains:
    - `raw/`: Downloaded and extracted data
    - `processed/`: Preprocessed data ready for training
    - Trained model files (.pth)

- **Root Directory**:
    - `Dockerfile`: Docker configuration with Python 3.10 and all dependencies.
    - `requirements.txt`: PyTorch, pandas, scikit-learn, and other required packages.
    - `run.sh`: Shell script that executes the full pipeline.
    - `README.md`: Project documentation and instructions.
