# Wind Speed Prediction Using LSTM Neural Networks
This project focuses on predicting wind speed using hybrid model: Convolutional Neural Network (CNN) with Long Short-Term Memory (LSTM) neural networks implemented in PyTorch. Accurate wind speed forecasting is essential for optimizing wind energy generation and ensuring the stability of power systems.

Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [References](#references)
## Introduction
Wind energy is a rapidly growing renewable energy source. However, the inherent variability and unpredictability of wind pose challenges for efficient power generation. Reliable short-term wind speed forecasts are crucial for the dynamic control of wind turbines and effective power system scheduling. In this project, we employ LSTM neural networks to capture temporal patterns in wind speed data, aiming to improve forecasting accuracy.

## Dataset
The dataset used in this project comprises time-series wind speed measurements collected over a specific period. The data includes features such as:

Date/Time: Timestamp of the observation
Wind Speed: Measured wind speed (m/s)
Additional Features: Other relevant meteorological variables (e.g., temperature, humidity)
Note: Ensure that the dataset is preprocessed to handle missing values and is normalized for optimal model performance.

## Installation
To set up the project environment, follow these steps:

1. **Clone the repository:**

'git clone https://github.com/thaianh1210/WindSpeedPrediction.git
cd WindSpeedPrediction'
2. **Create and activate a virtual environment:**

'python -m venv env
source env/bin/activate  # On Windows, use 'env\Scripts\activate''
3. **Install the required dependencies:**

'pip install -r requirements.txt'
The requirements.txt file includes:

PyQt5
plotly
pandas
numpy
scikit-learn
matplotlib
torch
torchvision
torchaudio
torchmetrics
torchinfo

## Usage
To run the wind speed prediction model:

Prepare the dataset: Ensure your dataset is in the correct format and located in the appropriate directory.

Execute the main script: Run the Jupyter Notebook Main_Advanced.ipynb to preprocess the data, train the model, and evaluate its performance.

'jupyter notebook Main_Advanced.ipynb'
Visualize the results: The notebook includes sections for data visualization, model training, and performance metrics.

## Model Architecture
The LSTM model is designed to capture temporal dependencies in wind speed data. The architecture includes:

Input Layer: Accepts sequences of wind speed measurements.
LSTM Layers: One or more LSTM layers to learn temporal patterns.
Fully Connected Layer: Maps the LSTM outputs to the desired prediction.
Output Layer: Produces the forecasted wind speed.
Hyperparameters such as the number of LSTM layers, hidden units, learning rate, and batch size can be adjusted to optimize performance.

## Results
The model's performance is evaluated using metrics like Mean Squared Error (MSE) and R-squared (R²). Visualization tools such as Plotly are employed to compare the predicted wind speeds against actual measurements, providing insights into the model's accuracy.

## References
Wind Speed Prediction using LSTMs in PyTorch
Time Series Analysis using LSTM for Wind Energy Prediction
Wind Speed Prediction Using Deep Learning-LSTM and GRU
For more details, refer to the Main_Advanced.ipynb notebook in this repository.
