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
warnings.filterwarnings('ignore')

class MultivariateLSTM(nn.Module):
    """
    Multivariate LSTM model for time series forecasting
    """
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 output_size: int, dropout: float = 0.2):
        super(MultivariateLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Use the last output for prediction
        out = self.fc(lstm_out[:, -1, :])
        return out

class HorticulturalSalesPredictor:
    """
    Main class for horticultural sales prediction using LSTM
    """
    def __init__(self, sequence_length: int = 12, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2, 
                 learning_rate: float = 0.001, batch_size: int = 32):
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()
        
        # Model and device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        
    def prepare_data(self, data: pd.DataFrame, target_column: str, 
                    train_percentage: float = 0.5) -> Tuple[np.ndarray, np.ndarray, 
                                                           np.ndarray, np.ndarray]:
        """
        Prepare data for training and validation
        """
        # Separate features and target
        features = data.drop(columns=[target_column]).values
        target = data[target_column].values.reshape(-1, 1)
        
        # Scale features and target
        features_scaled = self.feature_scaler.fit_transform(features)
        target_scaled = self.target_scaler.fit_transform(target)
        
        # Calculate split point
        total_points = len(data)
        train_points = int(total_points * train_percentage)
        
        print(f"Total time points: {total_points}")
        print(f"Training points: {train_points}")
        print(f"Validation points: {total_points - train_points}")
        
        # Split data
        X_train_scaled = features_scaled[:train_points]
        y_train_scaled = target_scaled[:train_points]
        X_val_scaled = features_scaled[train_points:]
        y_val_scaled = target_scaled[train_points:]
        
        return X_train_scaled, y_train_scaled, X_val_scaled, y_val_scaled
    
    def create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM input
        """
        X_seq, y_seq = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_seq.append(X[i:i + self.sequence_length])
            y_seq.append(y[i + self.sequence_length])
        
        return np.array(X_seq), np.array(y_seq)
    
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_val: np.ndarray, y_val: np.ndarray, 
                   epochs: int = 100, patience: int = 10) -> Dict:
        """
        Train the LSTM model with early stopping
        """
        # Create sequences
        X_train_seq, y_train_seq = self.create_sequences(X_train, y_train)
        X_val_seq, y_val_seq = self.create_sequences(X_val, y_val)
        
        # Check for NaN values and handle them
        if np.isnan(X_train_seq).any() or np.isnan(y_train_seq).any():
            print("Warning: NaN values found in training data. Removing NaN samples...")
            # Remove sequences with NaN values
            valid_indices = ~(np.isnan(X_train_seq).any(axis=(1,2)) | np.isnan(y_train_seq).any(axis=1))
            X_train_seq = X_train_seq[valid_indices]
            y_train_seq = y_train_seq[valid_indices]
            
        if np.isnan(X_val_seq).any() or np.isnan(y_val_seq).any():
            print("Warning: NaN values found in validation data. Removing NaN samples...")
            valid_indices = ~(np.isnan(X_val_seq).any(axis=(1,2)) | np.isnan(y_val_seq).any(axis=1))
            X_val_seq = X_val_seq[valid_indices]
            y_val_seq = y_val_seq[valid_indices]
        
        if len(X_train_seq) == 0 or len(X_val_seq) == 0:
            raise ValueError("No valid training or validation sequences after removing NaN values!")
        
        print(f"Training sequences: {len(X_train_seq)}, Validation sequences: {len(X_val_seq)}")
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(self.device)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(self.device)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Initialize model
        input_size = X_train_seq.shape[2]
        output_size = y_train_seq.shape[1]
        
        self.model = MultivariateLSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=output_size,
            dropout=self.dropout
        ).to(self.device)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop with early stopping
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        model_saved = False
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = self.model(batch_X)
                loss = criterion(output, batch_y)
                
                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss detected at epoch {epoch}, batch {batch_count}")
                    continue
                    
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                train_loss += loss.item()
                batch_count += 1
            
            if batch_count == 0:
                print(f"No valid batches in epoch {epoch}, stopping training")
                break
                
            train_loss /= batch_count
            train_losses.append(train_loss)
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_output = self.model(X_val_tensor)
                val_loss = criterion(val_output, y_val_tensor).item()
                
                # Check for NaN validation loss
                if np.isnan(val_loss):
                    print(f"Warning: NaN validation loss at epoch {epoch}")
                    val_loss = float('inf')
                    
                val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss and not np.isnan(val_loss):
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
                model_saved = True
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        # Load best model if it was saved
        if model_saved:
            self.model.load_state_dict(torch.load('best_model.pth'))
            print(f"Loaded best model with validation loss: {best_val_loss:.4f}")
        else:
            print("Warning: No valid model was saved due to NaN losses. Using last model state.")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        X_seq, _ = self.create_sequences(X, np.zeros((len(X), 1)))
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            predictions_scaled = self.model(X_tensor).cpu().numpy()
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions_scaled)
        return predictions
    
    def cross_validate(self, data: pd.DataFrame, target_column: str, 
                      train_percentage: float = 0.5, k_folds: int = 3,
                      epochs: int = 50) -> Dict:
        """
        Perform cross-validation on the validation set
        """
        # Prepare initial data split
        X_train, y_train, X_val, y_val = self.prepare_data(data, target_column, train_percentage)
        
        # Calculate fold size for validation set
        val_size = len(X_val)
        fold_size = val_size // k_folds
        horizon = fold_size
        
        print(f"\nCross-validation setup:")
        print(f"Validation set size: {val_size}")
        print(f"Number of folds: {k_folds}")
        print(f"Fold size (horizon): {fold_size}")
        
        cv_results = {
            'fold_scores': [],
            'fold_predictions': [],
            'fold_actuals': []
        }
        
        for fold in range(k_folds):
            print(f"\nTraining fold {fold + 1}/{k_folds}")
            
            # Define validation fold indices
            val_start = fold * fold_size
            val_end = min((fold + 1) * fold_size, val_size)
            
            # For each fold, use training data + previous validation folds as training
            if fold == 0:
                # First fold: use only original training data
                fold_X_train = X_train
                fold_y_train = y_train
            else:
                # Subsequent folds: add previous validation folds to training
                prev_val_X = X_val[:val_start]
                prev_val_y = y_val[:val_start]
                fold_X_train = np.vstack([X_train, prev_val_X])
                fold_y_train = np.vstack([y_train, prev_val_y])
            
            # Current validation fold
            fold_X_val = X_val[val_start:val_end]
            fold_y_val = y_val[val_start:val_end]
            
            # Train model for this fold
            training_history = self.train_model(
                fold_X_train, fold_y_train, fold_X_val, fold_y_val, epochs=epochs
            )
            
            # Make predictions
            predictions = self.predict(fold_X_val)
            actuals = self.target_scaler.inverse_transform(fold_y_val[self.sequence_length:])
            
            # Calculate metrics
            mse = mean_squared_error(actuals, predictions)
            mae = mean_absolute_error(actuals, predictions)
            r2 = r2_score(actuals, predictions)
            
            fold_score = {
                'fold': fold + 1,
                'mse': mse,
                'mae': mae,
                'r2': r2,
                'rmse': np.sqrt(mse)
            }
            
            cv_results['fold_scores'].append(fold_score)
            cv_results['fold_predictions'].append(predictions)
            cv_results['fold_actuals'].append(actuals)
            
            print(f"Fold {fold + 1} - MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['mse', 'mae', 'r2', 'rmse']:
            avg_metrics[f'avg_{metric}'] = np.mean([score[metric] for score in cv_results['fold_scores']])
            avg_metrics[f'std_{metric}'] = np.std([score[metric] for score in cv_results['fold_scores']])
        
        cv_results['average_metrics'] = avg_metrics
        
        print(f"\nCross-validation Results:")
        for metric in ['mse', 'mae', 'r2', 'rmse']:
            print(f"{metric.upper()}: {avg_metrics[f'avg_{metric}']:.4f} ± {avg_metrics[f'std_{metric}']:.4f}")
        
        return cv_results
    
    def plot_results(self, cv_results: Dict):
        """
        Plot cross-validation results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Predictions vs Actuals for all folds
        axes[0, 0].set_title('Predictions vs Actuals (All Folds)')
        for i, (pred, actual) in enumerate(zip(cv_results['fold_predictions'], 
                                             cv_results['fold_actuals'])):
            axes[0, 0].scatter(actual, pred, alpha=0.6, label=f'Fold {i+1}')
        
        # Perfect prediction line
        all_actuals = np.concatenate(cv_results['fold_actuals'])
        min_val, max_val = all_actuals.min(), all_actuals.max()
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 0].set_xlabel('Actual Sales')
        axes[0, 0].set_ylabel('Predicted Sales')
        axes[0, 0].legend()
        
        # Plot 2: Metrics across folds
        metrics = ['mse', 'mae', 'r2']
        fold_numbers = [score['fold'] for score in cv_results['fold_scores']]
        
        for i, metric in enumerate(metrics):
            if i < 2:
                ax = axes[0, 1] if i == 0 else axes[1, 0]
            else:
                ax = axes[1, 1]
            
            values = [score[metric] for score in cv_results['fold_scores']]
            ax.bar(fold_numbers, values, alpha=0.7)
            ax.set_title(f'{metric.upper()} across Folds')
            ax.set_xlabel('Fold')
            ax.set_ylabel(metric.upper())
            
            # Add average line
            avg_val = cv_results['average_metrics'][f'avg_{metric}']
            ax.axhline(y=avg_val, color='r', linestyle='--', alpha=0.8, 
                      label=f'Average: {avg_val:.4f}')
            ax.legend()
        
        plt.tight_layout()
        plt.show()

# Data loading and preprocessing functions
def load_tulip_data(file_path: str = 'OwnDoc.csv') -> pd.DataFrame:
    """
    Load and preprocess the tulip sales dataset
    """
    # Read the CSV file
    data = pd.read_csv(file_path, delimiter=';')
    
    # Convert Date to datetime
    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')
    
    # Handle decimal separator (replace comma with dot for numeric columns)
    numeric_columns = ['mean_temp', 'mean_humid', 'mean_prec_height_mm', 'total_prec_height_mm',
                      'mean_sun_dur_min', 'total_sun_dur_h']
    
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
    
    # Convert boolean-like columns
    bool_columns = ['mean_prec_flag', 'total_prec_flag']
    for col in bool_columns:
        if col in data.columns:
            data[col] = data[col].map({'True': 1, 'False': 0, True: 1, False: 0})
    
    # Convert categorical columns to numeric
    data['public_holiday'] = data['public_holiday'].map({'yes': 1, 'no': 0})
    data['school_holiday'] = data['school_holiday'].map({'yes': 1, 'no': 0})
    
    # Sort by date to ensure chronological order
    data = data.sort_values('Date').reset_index(drop=True)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {data.shape}")
    print(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"Columns: {list(data.columns)}")
    
    # Check for missing values
    missing_values = data.isnull().sum()
    if missing_values.any():
        print(f"\nMissing values found:")
        print(missing_values[missing_values > 0])
    
    return data

def prepare_tulip_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for modeling (exclude Date and target variable)
    """
    # Create a copy of the data
    features_data = data.copy()
    
    # Add time-based features
    features_data['day_of_year'] = features_data['Date'].dt.dayofyear
    features_data['month'] = features_data['Date'].dt.month
    features_data['day_of_week'] = features_data['Date'].dt.dayofweek
    features_data['week_of_year'] = features_data['Date'].dt.isocalendar().week
    
    # Add seasonal features
    features_data['sin_day_of_year'] = np.sin(2 * np.pi * features_data['day_of_year'] / 365.25)
    features_data['cos_day_of_year'] = np.cos(2 * np.pi * features_data['day_of_year'] / 365.25)
    features_data['sin_week'] = np.sin(2 * np.pi * features_data['week_of_year'] / 52)
    features_data['cos_week'] = np.cos(2 * np.pi * features_data['week_of_year'] / 52)
    
    # Drop Date column as it's not needed for modeling
    features_data = features_data.drop(columns=['Date'])
    
    # Ensure all columns are numeric
    non_numeric = features_data.select_dtypes(exclude=[np.number]).columns
    if len(non_numeric) > 0:
        print(f"Warning: Non-numeric columns found: {list(non_numeric)}")
        print("These columns will be excluded from modeling.")
        features_data = features_data.select_dtypes(include=[np.number])
    
    return features_data

def analyze_tulip_data(data: pd.DataFrame):
    """
    Perform exploratory data analysis on tulip sales data
    """
    print("\n=== TULIP SALES DATA ANALYSIS ===")
    
    # Basic statistics
    print(f"\nBasic Statistics for SoldTulips:")
    print(f"Mean: {data['SoldTulips'].mean():.2f}")
    print(f"Median: {data['SoldTulips'].median():.2f}")
    print(f"Std: {data['SoldTulips'].std():.2f}")
    print(f"Min: {data['SoldTulips'].min()}")
    print(f"Max: {data['SoldTulips'].max()}")
    
    # Correlation analysis
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    correlations = data[numeric_cols].corr()['SoldTulips'].sort_values(ascending=False)
    
    print(f"\nCorrelations with SoldTulips (top 5 positive and negative):")
    print("Positive correlations:")
    print(correlations.head(6)[1:])  # Exclude self-correlation
    print("\nNegative correlations:")
    print(correlations.tail(5))
    
    return correlations

def main():
    """
    Main execution function for tulip sales prediction
    """
    print("=== TULIP SALES PREDICTION WITH LSTM ===\n")
    
    # Configuration
    config = {
        'file_path': 'Data/OwnDoc.csv',
        'target_column': 'SoldTulips',
        'train_percentage': 0.5,
        'k_folds': 3,
        'sequence_length': 7,  # Weekly patterns
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 100
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Load tulip data
        print("Loading tulip sales data...")
        raw_data = load_tulip_data(config['file_path'])
        
        # Analyze the data
        correlations = analyze_tulip_data(raw_data)
        
        # Prepare features
        print("\nPreparing features...")
        features_data = prepare_tulip_features(raw_data)
        print(f"Final feature set shape: {features_data.shape}")
        print(f"Features: {[col for col in features_data.columns if col != config['target_column']]}")
        
        # Check if target column exists
        if config['target_column'] not in features_data.columns:
            raise ValueError(f"Target column '{config['target_column']}' not found in data")
        
        # Initialize predictor
        predictor = HorticulturalSalesPredictor(
            sequence_length=config['sequence_length'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            learning_rate=config['learning_rate'],
            batch_size=config['batch_size']
        )
        
        # Perform cross-validation
        print(f"\nStarting cross-validation for {config['target_column']} prediction...")
        cv_results = predictor.cross_validate(
            data=features_data,
            target_column=config['target_column'],
            train_percentage=config['train_percentage'],
            k_folds=config['k_folds'],
            epochs=config['epochs']
        )
        
        # Plot results
        predictor.plot_results(cv_results)
        
        # Display summary
        print("\n" + "="*60)
        print(f"FINAL RESULTS SUMMARY - {config['target_column']} PREDICTION")
        print("="*60)
        
        avg_metrics = cv_results['average_metrics']
        print(f"Root Mean Square Error: {avg_metrics['avg_rmse']:.2f} ± {avg_metrics['std_rmse']:.2f} tulips")
        print(f"Mean Absolute Error: {avg_metrics['avg_mae']:.2f} ± {avg_metrics['std_mae']:.2f} tulips")
        print(f"R-squared Score: {avg_metrics['avg_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
        
        # Calculate percentage errors
        target_mean = features_data[config['target_column']].mean()
        mape = (avg_metrics['avg_mae'] / target_mean) * 100
        print(f"Mean Absolute Percentage Error: {mape:.2f}%")
        
        # Feature importance (based on correlations)
        print(f"\nTop 5 most correlated features with {config['target_column']}:")
        top_features = correlations.abs().sort_values(ascending=False)[1:6]  # Exclude self-correlation
        for feature, corr in top_features.items():
            print(f"  {feature}: {corr:.3f}")
        
        return predictor, cv_results, raw_data, features_data
        
    except FileNotFoundError:
        print(f"Error: Could not find the file '{config['file_path']}'")
        print("Please make sure the OwnDoc.csv file is in the same directory as this script.")
        return None, None, None, None
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None, None

if __name__ == "__main__":
    # Updated to match the 4 return values from main()
    predictor, results, raw_data, features_data = main()
