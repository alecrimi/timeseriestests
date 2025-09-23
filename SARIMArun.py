import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# SARIMA and statistical libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy import stats

# Metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Grid search for hyperparameter optimization
from itertools import product

class SARIMAPredictor:
    """
    SARIMA model for horticultural sales prediction
    """
    def __init__(self, seasonal_period: int = 7):
        """
        Initialize SARIMA predictor
        
        Args:
            seasonal_period: Number of periods in a season (e.g., 7 for weekly seasonality)
        """
        self.seasonal_period = seasonal_period
        self.model = None
        self.fitted_model = None
        self.best_params = None
        self.aic_score = None
        
    def check_stationarity(self, series: pd.Series, significance_level: float = 0.05) -> Dict:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        """
        result = adfuller(series.dropna())
        
        is_stationary = result[1] < significance_level
        
        return {
            'is_stationary': is_stationary,
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'interpretation': 'Stationary' if is_stationary else 'Non-stationary'
        }
    
    def make_stationary(self, series: pd.Series) -> Tuple[pd.Series, int, bool]:
        """
        Make time series stationary through differencing
        
        Returns:
            stationary_series, num_differences, seasonal_diff_applied
        """
        original_series = series.copy()
        current_series = series.copy()
        num_regular_diff = 0
        seasonal_diff_applied = False
        
        # First check if seasonal differencing is needed
        seasonal_test = self.check_stationarity(current_series)
        if not seasonal_test['is_stationary'] and len(current_series) > self.seasonal_period:
            # Try seasonal differencing first
            seasonal_diff = current_series.diff(self.seasonal_period).dropna()
            if len(seasonal_diff) > 0:
                seasonal_stationarity = self.check_stationarity(seasonal_diff)
                if seasonal_stationarity['p_value'] < seasonal_test['p_value']:
                    current_series = seasonal_diff
                    seasonal_diff_applied = True
                    print(f"Applied seasonal differencing (period={self.seasonal_period})")
        
        # Then apply regular differencing if needed
        max_diff = 3  # Prevent over-differencing
        while num_regular_diff < max_diff:
            stationarity_test = self.check_stationarity(current_series)
            
            if stationarity_test['is_stationary']:
                print(f"Series is stationary after {num_regular_diff} regular differences")
                break
            else:
                current_series = current_series.diff().dropna()
                num_regular_diff += 1
                print(f"Applied regular differencing #{num_regular_diff}")
        
        return current_series, num_regular_diff, seasonal_diff_applied
    
    def grid_search_sarima(self, series: pd.Series, max_p: int = 2, max_d: int = 1, max_q: int = 2,
                          max_P: int = 1, max_D: int = 1, max_Q: int = 1, 
                          information_criterion: str = 'aic') -> Dict:
        """
        Perform grid search to find optimal SARIMA parameters
        """
        n_obs = len(series)
        print(f"Starting SARIMA grid search...")
        print(f"Sample size: {n_obs}")
        
        # Adjust parameters based on sample size
        if n_obs < 50:
            max_p, max_q, max_P, max_Q = 1, 1, 0, 0  # Very conservative for small samples
            print("Small sample detected - using conservative parameters")
        elif n_obs < 100:
            max_p, max_q, max_P, max_Q = 2, 2, 1, 1
            print("Medium sample detected - using moderate parameters")
        
        # Check if we have enough data for seasonal modeling
        min_seasonal_obs = self.seasonal_period * 3  # Need at least 3 full seasons
        if n_obs < min_seasonal_obs:
            max_P, max_D, max_Q = 0, 0, 0  # No seasonal components
            self.seasonal_period = 1  # Disable seasonality
            print(f"Insufficient data for seasonal modeling ({n_obs} < {min_seasonal_obs}). Using non-seasonal ARIMA.")
        
        print(f"Parameter ranges: p(0-{max_p}), d(0-{max_d}), q(0-{max_q})")
        print(f"Seasonal ranges: P(0-{max_P}), D(0-{max_D}), Q(0-{max_Q}), S={self.seasonal_period}")
        
        best_score = np.inf
        best_params = None
        results = []
        successful_fits = 0
        
        # Generate all parameter combinations
        param_combinations = list(product(
            range(max_p + 1),  # p
            range(max_d + 1),  # d  
            range(max_q + 1),  # q
            range(max_P + 1),  # P
            range(max_D + 1),  # D
            range(max_Q + 1)   # Q
        ))
        
        print(f"Testing {len(param_combinations)} parameter combinations...")
        
        for i, (p, d, q, P, D, Q) in enumerate(param_combinations):
            try:
                order = (p, d, q)
                seasonal_order = (P, D, Q, self.seasonal_period) if self.seasonal_period > 1 else (0, 0, 0, 0)
                
                # Check minimum sample size requirements
                total_params = p + d + q + P + D + Q
                min_required_obs = total_params * 3 + self.seasonal_period
                
                if n_obs < min_required_obs:
                    continue
                
                # Skip if too many parameters (overfitting risk)
                if total_params > n_obs // 10:  # Conservative rule: max 1 param per 10 observations
                    continue
                
                # Skip problematic combinations
                if d + D > 2:  # Too much differencing
                    continue
                
                # Fit SARIMA model
                model = SARIMAX(series, 
                              order=order,
                              seasonal_order=seasonal_order,
                              enforce_stationarity=False,
                              enforce_invertibility=False,
                              simple_differencing=True)  # Add this for stability
                
                fitted_model = model.fit(disp=False, maxiter=50, method='lbfgs')
                
                # Check model convergence
                if not fitted_model.mle_retvals['converged']:
                    continue
                
                # Get information criterion score
                if information_criterion.lower() == 'aic':
                    score = fitted_model.aic
                elif information_criterion.lower() == 'bic':
                    score = fitted_model.bic
                else:
                    score = fitted_model.aic
                
                # Check for valid score
                if np.isnan(score) or np.isinf(score):
                    continue
                
                results.append({
                    'order': order,
                    'seasonal_order': seasonal_order,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic,
                    'score': score
                })
                
                successful_fits += 1
                
                if score < best_score:
                    best_score = score
                    best_params = {
                        'order': order,
                        'seasonal_order': seasonal_order,
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic
                    }
                
                if successful_fits % 5 == 0:
                    print(f"Successful fits: {successful_fits}/{i+1} tested. Best {information_criterion.upper()}: {best_score:.2f}")
                
            except Exception as e:
                # Skip failed model fits
                continue
        
        print(f"\nGrid search completed: {successful_fits} successful fits out of {len(param_combinations)} tested")
        
        if best_params is None:
            # Fallback to simple ARIMA models
            print("No SARIMA models converged. Trying simple ARIMA models...")
            return self._fallback_simple_arima(series, information_criterion)
        
        print(f"Best parameters: SARIMA{best_params['order']}x{best_params['seasonal_order']}")
        print(f"Best AIC: {best_params['aic']:.2f}")
        print(f"Best BIC: {best_params['bic']:.2f}")
        
        # Sort results by score for analysis
        results_df = pd.DataFrame(results).sort_values('score') if results else pd.DataFrame()
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results_df
        }
    
    def _fallback_simple_arima(self, series: pd.Series, information_criterion: str = 'aic') -> Dict:
        """
        Fallback method to try very simple ARIMA models when SARIMA fails
        """
        print("Trying fallback simple ARIMA models...")
        
        simple_orders = [
            (0, 1, 0),  # Random walk
            (1, 0, 0),  # AR(1)
            (0, 0, 1),  # MA(1)
            (1, 1, 0),  # ARIMA(1,1,0)
            (0, 1, 1),  # ARIMA(0,1,1)
            (1, 1, 1),  # ARIMA(1,1,1)
        ]
        
        best_score = np.inf
        best_params = None
        
        for order in simple_orders:
            try:
                model = SARIMAX(series, order=order, seasonal_order=(0, 0, 0, 0))
                fitted_model = model.fit(disp=False, maxiter=50)
                
                score = fitted_model.aic if information_criterion.lower() == 'aic' else fitted_model.bic
                
                if score < best_score and not np.isnan(score):
                    best_score = score
                    best_params = {
                        'order': order,
                        'seasonal_order': (0, 0, 0, 0),
                        'aic': fitted_model.aic,
                        'bic': fitted_model.bic
                    }
                    print(f"Simple ARIMA{order} - AIC: {fitted_model.aic:.2f}")
                    
            except Exception as e:
                continue
        
        if best_params is None:
            # Ultimate fallback: random walk
            best_params = {
                'order': (0, 1, 0),
                'seasonal_order': (0, 0, 0, 0),
                'aic': 999999,
                'bic': 999999
            }
            print("Using random walk model as ultimate fallback")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': pd.DataFrame()
        }
    
    def fit_model(self, series: pd.Series, order: Optional[Tuple] = None, 
                  seasonal_order: Optional[Tuple] = None, auto_search: bool = True) -> Dict:
        """
        Fit SARIMA model to the time series
        """
        if auto_search and (order is None or seasonal_order is None):
            print("Performing automatic parameter search...")
            search_results = self.grid_search_sarima(series)
            self.best_params = search_results['best_params']
            order = self.best_params['order']
            seasonal_order = self.best_params['seasonal_order']
        elif order is None or seasonal_order is None:
            # Default parameters if not provided
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, self.seasonal_period)
            print(f"Using default parameters: SARIMA{order}x{seasonal_order}")
        
        try:
            # Fit the model
            print(f"Fitting SARIMA{order}x{seasonal_order} model...")
            self.model = SARIMAX(series,
                               order=order,
                               seasonal_order=seasonal_order,
                               enforce_stationarity=False,
                               enforce_invertibility=False)
            
            self.fitted_model = self.model.fit(disp=False, maxiter=200)
            self.aic_score = self.fitted_model.aic
            
            print(f"Model fitted successfully!")
            print(f"AIC: {self.fitted_model.aic:.2f}")
            print(f"BIC: {self.fitted_model.bic:.2f}")
            print(f"Log-likelihood: {self.fitted_model.llf:.2f}")
            
            # Model diagnostics
            diagnostics = self.model_diagnostics()
            
            return {
                'model': self.fitted_model,
                'aic': self.fitted_model.aic,
                'bic': self.fitted_model.bic,
                'params': self.fitted_model.params,
                'diagnostics': diagnostics
            }
            
        except Exception as e:
            print(f"Error fitting SARIMA model: {str(e)}")
            raise
    
    def model_diagnostics(self) -> Dict:
        """
        Perform model diagnostics
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first!")
        
        residuals = self.fitted_model.resid
        
        # Ljung-Box test for residual autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//5), return_df=True)
        
        # Normality test for residuals
        normality_stat, normality_pvalue = stats.jarque_bera(residuals.dropna())
        
        diagnostics = {
            'ljung_box_pvalue': lb_test['lb_pvalue'].iloc[-1] if len(lb_test) > 0 else None,
            'normality_pvalue': normality_pvalue,
            'residuals_mean': residuals.mean(),
            'residuals_std': residuals.std(),
            'residuals_skewness': stats.skew(residuals.dropna()),
            'residuals_kurtosis': stats.kurtosis(residuals.dropna())
        }
        
        print(f"\nModel Diagnostics:")
        print(f"Ljung-Box p-value: {diagnostics['ljung_box_pvalue']:.4f}" if diagnostics['ljung_box_pvalue'] else "Ljung-Box test failed")
        print(f"Residuals normality p-value: {diagnostics['normality_pvalue']:.4f}")
        print(f"Residuals mean: {diagnostics['residuals_mean']:.4f}")
        print(f"Residuals std: {diagnostics['residuals_std']:.4f}")
        
        return diagnostics
    
    def predict(self, steps: int = 1, return_conf_int: bool = True) -> Dict:
        """
        Make predictions using fitted SARIMA model
        """
        if self.fitted_model is None:
            raise ValueError("Model must be fitted first!")
        
        # Get forecast
        forecast_result = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast_result.predicted_mean
        
        result = {
            'predictions': predictions,
            'steps': steps
        }
        
        if return_conf_int:
            conf_int = forecast_result.conf_int()
            result['conf_int_lower'] = conf_int.iloc[:, 0]
            result['conf_int_upper'] = conf_int.iloc[:, 1]
        
        return result
    
    def cross_validate(self, series: pd.Series, train_percentage: float = 0.5, 
                      k_folds: int = 3, auto_search: bool = True) -> Dict:
        """
        Perform walk-forward cross-validation
        """
        print(f"\n=== SARIMA Cross-Validation ===")
        print(f"Train percentage: {train_percentage}")
        print(f"Number of folds: {k_folds}")
        
        # Calculate split points
        total_points = len(series)
        train_size = int(total_points * train_percentage)
        test_size = total_points - train_size
        fold_size = test_size // k_folds
        
        print(f"Total points: {total_points}")
        print(f"Training points: {train_size}")
        print(f"Test points: {test_size}")
        print(f"Fold size: {fold_size}")
        
        cv_results = {
            'fold_scores': [],
            'fold_predictions': [],
            'fold_actuals': [],
            'fold_models': []
        }
        
        # Get optimal parameters from full training set if auto_search is enabled
        if auto_search:
            print("\nFinding optimal parameters using full training set...")
            train_series = series.iloc[:train_size]
            search_results = self.grid_search_sarima(train_series)
            optimal_order = search_results['best_params']['order']
            optimal_seasonal_order = search_results['best_params']['seasonal_order']
            print(f"Optimal parameters: SARIMA{optimal_order}x{optimal_seasonal_order}")
        else:
            optimal_order = (1, 1, 1)
            optimal_seasonal_order = (1, 1, 1, self.seasonal_period)
        
        for fold in range(k_folds):
            print(f"\n--- Fold {fold + 1}/{k_folds} ---")
            
            # Define fold boundaries
            test_start = train_size + fold * fold_size
            test_end = min(train_size + (fold + 1) * fold_size, total_points)
            
            # Training data includes original training set plus any previous test folds
            fold_train_end = test_start
            fold_train_series = series.iloc[:fold_train_end]
            fold_test_series = series.iloc[test_start:test_end]
            
            print(f"Train data: index 0 to {fold_train_end-1} ({len(fold_train_series)} points)")
            print(f"Test data: index {test_start} to {test_end-1} ({len(fold_test_series)} points)")
            
            try:
                # Fit model for this fold
                fold_predictor = SARIMAPredictor(seasonal_period=self.seasonal_period)
                fold_predictor.fit_model(fold_train_series, 
                                       order=optimal_order,
                                       seasonal_order=optimal_seasonal_order,
                                       auto_search=False)
                
                # Make predictions
                n_steps = len(fold_test_series)
                predictions_dict = fold_predictor.predict(steps=n_steps, return_conf_int=True)
                predictions = predictions_dict['predictions'].values
                
                # Calculate metrics
                actuals = fold_test_series.values
                mse = mean_squared_error(actuals, predictions)
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(actuals, predictions) if len(np.unique(actuals)) > 1 else 0.0
                
                fold_score = {
                    'fold': fold + 1,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'aic': fold_predictor.fitted_model.aic,
                    'n_train': len(fold_train_series),
                    'n_test': len(fold_test_series)
                }
                
                cv_results['fold_scores'].append(fold_score)
                cv_results['fold_predictions'].append(predictions)
                cv_results['fold_actuals'].append(actuals)
                cv_results['fold_models'].append(fold_predictor.fitted_model)
                
                print(f"Fold {fold + 1} Results:")
                print(f"  RMSE: {rmse:.2f}")
                print(f"  MAE: {mae:.2f}")
                print(f"  R²: {r2:.4f}")
                print(f"  AIC: {fold_predictor.fitted_model.aic:.2f}")
                
            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                continue
        
        if not cv_results['fold_scores']:
            raise ValueError("All folds failed. Check your data and parameters.")
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in ['mse', 'mae', 'rmse', 'r2', 'aic']:
            values = [score[metric] for score in cv_results['fold_scores']]
            avg_metrics[f'avg_{metric}'] = np.mean(values)
            avg_metrics[f'std_{metric}'] = np.std(values)
        
        cv_results['average_metrics'] = avg_metrics
        cv_results['optimal_params'] = {
            'order': optimal_order,
            'seasonal_order': optimal_seasonal_order
        }
        
        print(f"\n=== Cross-Validation Summary ===")
        print(f"RMSE: {avg_metrics['avg_rmse']:.2f} ± {avg_metrics['std_rmse']:.2f}")
        print(f"MAE: {avg_metrics['avg_mae']:.2f} ± {avg_metrics['std_mae']:.2f}")
        print(f"R²: {avg_metrics['avg_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
        print(f"AIC: {avg_metrics['avg_aic']:.2f} ± {avg_metrics['std_aic']:.2f}")
        
        return cv_results
    
    def plot_results(self, series: pd.Series, cv_results: Dict):
        """
        Plot cross-validation results and diagnostics
        """
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot 1: Time series with train/test splits
        axes[0, 0].plot(series.index, series.values, 'b-', alpha=0.7, label='Actual')
        
        # Add fold boundaries
        total_points = len(series)
        train_size = int(total_points * 0.5)  # Assuming 50% train split
        axes[0, 0].axvline(x=series.index[train_size], color='red', linestyle='--', alpha=0.8, label='Train/Test Split')
        
        axes[0, 0].set_title('Time Series with Cross-Validation Splits')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Sales')
        axes[0, 0].legend()
        
        # Plot 2: Predictions vs Actuals
        all_predictions = np.concatenate(cv_results['fold_predictions'])
        all_actuals = np.concatenate(cv_results['fold_actuals'])
        
        axes[0, 1].scatter(all_actuals, all_predictions, alpha=0.6)
        min_val, max_val = all_actuals.min(), all_actuals.max()
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_xlabel('Actual Sales')
        axes[0, 1].set_ylabel('Predicted Sales')
        axes[0, 1].set_title('Predictions vs Actuals (All Folds)')
        
        # Plot 3: Metrics across folds
        metrics = ['rmse', 'mae', 'r2']
        fold_numbers = [score['fold'] for score in cv_results['fold_scores']]
        
        for i, metric in enumerate(metrics):
            values = [score[metric] for score in cv_results['fold_scores']]
            axes[0, 2].bar([f + i*0.25 for f in fold_numbers], values, 
                          width=0.25, alpha=0.7, label=metric.upper())
        
        axes[0, 2].set_xlabel('Fold')
        axes[0, 2].set_ylabel('Metric Value')
        axes[0, 2].set_title('Metrics Across Folds')
        axes[0, 2].legend()
        
        # Plot 4: Residuals from last fold
        if cv_results['fold_models']:
            last_model = cv_results['fold_models'][-1]
            residuals = last_model.resid
            axes[1, 0].plot(residuals.index, residuals.values)
            axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[1, 0].set_title('Residuals (Last Fold)')
            axes[1, 0].set_xlabel('Time')
            axes[1, 0].set_ylabel('Residuals')
        
        # Plot 5: Residuals histogram
        if cv_results['fold_models']:
            residuals = cv_results['fold_models'][-1].resid.dropna()
            axes[1, 1].hist(residuals, bins=20, alpha=0.7, density=True)
            axes[1, 1].set_title('Residuals Distribution')
            axes[1, 1].set_xlabel('Residual Value')
            axes[1, 1].set_ylabel('Density')
            
            # Overlay normal distribution
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            y = stats.norm.pdf(x, mu, sigma)
            axes[1, 1].plot(x, y, 'r-', alpha=0.8, label='Normal')
            axes[1, 1].legend()
        
        # Plot 6: Model performance summary
        avg_metrics = cv_results['average_metrics']
        metric_names = ['RMSE', 'MAE', 'R²']
        metric_values = [avg_metrics['avg_rmse'], avg_metrics['avg_mae'], avg_metrics['avg_r2']]
        metric_errors = [avg_metrics['std_rmse'], avg_metrics['std_mae'], avg_metrics['std_r2']]
        
        bars = axes[1, 2].bar(metric_names, metric_values, yerr=metric_errors, 
                             capsize=5, alpha=0.7, color=['blue', 'green', 'orange'])
        axes[1, 2].set_title('Average Performance Metrics')
        axes[1, 2].set_ylabel('Metric Value')
        
        # Add value labels on bars
        for bar, value, error in zip(bars, metric_values, metric_errors):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height,
                           f'{value:.3f}±{error:.3f}',
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

# Data loading and preprocessing functions (same as LSTM version)
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
    
    # Time series analysis
    sales_series = pd.Series(data['SoldTulips'].values, index=data['Date'])
    
    # Check for seasonality
    print(f"\nTime Series Properties:")
    print(f"Frequency: Daily")
    print(f"Total observations: {len(sales_series)}")
    
    # Seasonal decomposition if enough data
    if len(sales_series) > 14:  # Need at least 2 weeks for weekly seasonality
        try:
            decomposition = seasonal_decompose(sales_series, model='additive', period=7)
            print(f"Seasonal decomposition completed (weekly pattern)")
        except:
            print(f"Seasonal decomposition failed - may need more data or different period")
    
    return sales_series

def main():
    """
    Main execution function for SARIMA tulip sales prediction
    """
    print("=== TULIP SALES PREDICTION WITH SARIMA ===\n")
    
    # Configuration
    config = {
        'file_path': 'Data/OwnDoc.csv',
        'target_column': 'SoldTulips',
        'train_percentage': 0.5,
        'k_folds': 3,
        'seasonal_period': 7,  # Weekly seasonality
        'auto_search': True,   # Automatic parameter search
        'max_p': 3, 'max_d': 2, 'max_q': 3,  # ARIMA parameters
        'max_P': 2, 'max_D': 1, 'max_Q': 2   # Seasonal parameters
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    try:
        # Load tulip data
        print("Loading tulip sales data...")
        raw_data = load_tulip_data(config['file_path'])
        
        # Analyze the data and get time series
        sales_series = analyze_tulip_data(raw_data)
        
        # Check stationarity
        predictor = SARIMAPredictor(seasonal_period=config['seasonal_period'])
        stationarity_test = predictor.check_stationarity(sales_series)
        
        print(f"\nStationarity Test Results:")
        print(f"Is stationary: {stationarity_test['is_stationary']}")
        print(f"ADF p-value: {stationarity_test['p_value']:.4f}")
        print(f"Interpretation: {stationarity_test['interpretation']}")
        
        # Perform cross-validation
        print(f"\nStarting SARIMA cross-validation...")
        cv_results = predictor.cross_validate(
            series=sales_series,
            train_percentage=config['train_percentage'],
            k_folds=config['k_folds'],
            auto_search=config['auto_search']
        )
        
        # Plot results
        predictor.plot_results(sales_series, cv_results)
        
        # Display summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY - SARIMA MODEL")
        print("="*60)
        
        avg_metrics = cv_results['average_metrics']
        print(f"Root Mean Square Error: {avg_metrics['avg_rmse']:.2f} ± {avg_metrics['std_rmse']:.2f} tulips")
        print(f"Mean Absolute Error: {avg_metrics['avg_mae']:.2f} ± {avg_metrics['std_mae']:.2f} tulips")
        print(f"R-squared Score: {avg_metrics['avg_r2']:.4f} ± {avg_metrics['std_r2']:.4f}")
        print(f"Average AIC: {avg_metrics['avg_aic']:.2f} ± {avg_metrics['std_aic']:.2f}")
        
        # Calculate percentage errors
        target_mean = sales_series.mean()
        mape = (avg_metrics['avg_mae'] / target_mean) * 100
        print(f"Mean Absolute Percentage Error: {mape:.
