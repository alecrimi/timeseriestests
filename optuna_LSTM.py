import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
import yaml
import json
import optuna

warnings.filterwarnings('ignore')

# ======================================================
# Your original classes (unchanged)
# ======================================================
class MultivariateLSTM(nn.Module):
    """Multivariate LSTM model for time series forecasting"""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out


class HorticulturalSalesPredictor:
    """Main class for horticultural sales prediction using LSTM"""
    def __init__(self, sequence_length: int = 12, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 learning_rate: float = 0.001, batch_size: int = 32):
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
    # -----------------------------
    # keep your prepare_data, create_sequences, train_model,
    # predict, cross_validate, plot_results methods here (unchanged)
    # -----------------------------
    # (I won't paste them again to save space — keep them exactly as in your code)


# ======================================================
# Data loading and feature engineering (unchanged)
# ======================================================
def load_tulip_data(file_path: str = 'OwnDoc.csv') -> pd.DataFrame:
    # your implementation (unchanged)
    ...

def prepare_tulip_features(data: pd.DataFrame) -> pd.DataFrame:
    # your implementation (unchanged)
    ...

def analyze_tulip_data(data: pd.DataFrame):
    # your implementation (unchanged)
    ...


# ======================================================
# Optuna integration
# ======================================================
def train_and_evaluate(params, raw_data: pd.DataFrame, target_column: str):
    """Wrapper that uses your predictor with given params"""
    features_data = prepare_tulip_features(raw_data)

    predictor = HorticulturalSalesPredictor(
        sequence_length=params["sequence_length"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        learning_rate=params["learning_rate"],
        batch_size=params["batch_size"]
    )
    cv_results = predictor.cross_validate(
        data=features_data,
        target_column=target_column,
        train_percentage=params["train_percentage"],
        k_folds=params["k_folds"],
        epochs=params["epochs"]
    )
    return cv_results["average_metrics"]["avg_rmse"]  # minimize RMSE


def objective(trial, raw_data, target_column):
    params = {
        "sequence_length": trial.suggest_categorical("sequence_length", [7, 14, 21]),
        "hidden_size": trial.suggest_categorical("hidden_size", [32, 64, 128]),
        "num_layers": trial.suggest_int("num_layers", 1, 3),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        "learning_rate": trial.suggest_categorical("learning_rate", [1e-4, 5e-4, 1e-3]),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "epochs": trial.suggest_categorical("epochs", [50, 100]),
        "train_percentage": 0.5,
        "k_folds": 3
    }
    return train_and_evaluate(params, raw_data, target_column)


# ======================================================
# Main function
# ======================================================
def main():
    print("=== TULIP SALES PREDICTION WITH LSTM ===\n")

    # Load config.yaml
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()

    raw_data = load_tulip_data(config["file_path"])
    analyze_tulip_data(raw_data)

    if config.get("optuna_search", False):
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: objective(trial, raw_data, config["target_column"]),
                       n_trials=config.get("n_trials", 10))

        print("Best trial:", study.best_trial.params)
        with open("optimized_config.json", "w") as f:
            json.dump(study.best_trial.params, f, indent=2)
        return None, None, raw_data, None
    else:
        # fixed run (your original workflow)
        features_data = prepare_tulip_features(raw_data)
        predictor = HorticulturalSalesPredictor(
            sequence_length=config["sequence_length"],
            hidden_size=config["hidden_size"],
            num_layers=config["num_layers"],
            dropout=config["dropout"],
            learning_rate=config["learning_rate"],
            batch_size=config["batch_size"]
        )
        cv_results = predictor.cross_validate(
            data=features_data,
            target_column=config["target_column"],
            train_percentage=config["train_percentage"],
            k_folds=config["k_folds"],
            epochs=config["epochs"]
        )
        predictor.plot_results(cv_results)
        return predictor, cv_results, raw_data, features_data


if __name__ == "__main__":
    predictor, results, raw_data, features_data = main()
