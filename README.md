# LSTM_MSc_Thesis
This repo contains two main scripts:
### lstm.py -> python script version of the initial notebook
Functions: load_series, run_grid, train_final
### run_hourly.py
Small runner that loads data from Excel, does a tiny grid search, then trains and saves outputs.
### File: lstm_.py
What it does

Builds a 2-layer LSTM and trains with MSE.
Outputs:
model.keras – trained Keras model
scaler.joblib – fitted MinMaxScaler
predictions.csv – date, y_true, y_pred
predictions.png – quick plot
### File: run_hourly.py
Usage
python run_hourly.py 

Arguments

--input (path): Excel file

--sheet (str): worksheet name (e.g., H28)

--date-col (str): datetime column (e.g., Timestamp)

--target-col (str): numeric target column (e.g., H28)

--out-dir (path): output folder

--lookbacks (csv/space list): e.g., "48,72,96" or "720"

--train-size (float 0–1): e.g., 0.8

--epochs (int)

--batch (int)

--title (str): plot title

What it does

Reads the Excel sheet into a Pandas Series (indexed by date).

Runs a small grid search over your lookbacks/epochs/batch.

Trains the final model and writes the artifacts.
