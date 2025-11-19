from pyspark.sql import functions as F
from pyspark.sql.window import Window
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType
from datetime import datetime, timedelta

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# Select specific product designation
specific_product = "your_specific_product"  # Change this to your actual product designation

df_specific_product = df.filter(df["ProductDesignation"] == specific_product)

print(f"Number of records for product '{specific_product}': {df_specific_product.count()}")

# Identify numeric columns and target variable
target_column = "AVAFailure"

# Check if there's a timestamp/date column in your data
# Adjust the column name based on your actual schema
timestamp_col = "timestamp"  # Change this to your actual timestamp column name

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
                and f.name != target_column 
                and f.name != "ProductDesignation"
                and f.name != "part_id"
                and not f.name.startswith("id")]

print(f"Numeric features selected: {numeric_cols}")

# Weekly aggregation using PySpark
print("\nPerforming weekly aggregation...")

# Add week column (week of year)
df_with_week = df_specific_product.withColumn(
    "week", 
    F.weekofyear(F.col(timestamp_col))
).withColumn(
    "year",
    F.year(F.col(timestamp_col))
).withColumn(
    "year_week",
    F.concat(F.col("year"), F.lit("-W"), F.lpad(F.col("week"), 2, "0"))
)

# Aggregate by week - calculate mean for numeric columns
agg_expressions = [F.mean(col).alias(col) for col in numeric_cols + [target_column]]
agg_expressions.append(F.first("year_week").alias("year_week"))

df_weekly = df_with_week.groupBy("year", "week").agg(*agg_expressions)

# Sort by year and week to maintain temporal order
df_weekly = df_weekly.orderBy("year", "week")

print(f"Number of weeks after aggregation: {df_weekly.count()}")

# Convert to pandas for time series processing
pandas_df = df_weekly.select(numeric_cols + [target_column, "year_week"]).toPandas()

if len(pandas_df) == 0:
    raise ValueError(f"No data found for product designation: {specific_product}")
    
print(f"Data shape after weekly aggregation for {specific_product}: {pandas_df.shape}")
print(f"Date range: {pandas_df['year_week'].iloc[0]} to {pandas_df['year_week'].iloc[-1]}")

def create_sequences(data, features, target, sequence_length=10):
    """Create sequences for LSTM training"""
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[features].iloc[i:(i + sequence_length)].values)
        y.append(data[target].iloc[i + sequence_length])
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, output_dim=1, problem_type='regression'):
    """Build LSTM model for time series analysis"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(output_dim, activation='linear' if problem_type == 'regression' else 'softmax')
    ])
    
    if problem_type == 'regression':
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mse', metrics=['mae'])
    else:
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Prepare data for LSTM
sequence_length = 10

# Check target variable characteristics
target_stats = pandas_df[target_column].describe()
print(f"\nTarget variable '{target_column}' statistics:")
print(target_stats)

# Determine problem type based on target variable
unique_targets = pandas_df[target_column].nunique()
problem_type = 'classification' if unique_targets <= 10 else 'regression'
print(f"Problem type: {problem_type} (unique values: {unique_targets})")

# Scale features and prepare sequences
scaler = StandardScaler()
scaled_features = scaler.fit_transform(pandas_df[numeric_cols])
scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
scaled_df[target_column] = pandas_df[target_column].values
scaled_df['year_week'] = pandas_df['year_week'].values

# Create sequences
if len(scaled_df) > sequence_length:
    X_sequences, y_sequences = create_sequences(scaled_df, numeric_cols, target_column, sequence_length)
    
    print(f"\nSequences created - X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")
    
    # Split into train (70%) and test (30%) - temporal split
    n_samples = len(X_sequences)
    train_size = int(n_samples * 0.7)
    
    X_train = X_sequences[:train_size]
    y_train = y_sequences[:train_size]
    X_test = X_sequences[train_size:]
    y_test = y_sequences[train_size:]
    
    print(f"\nTrain set: {X_train.shape[0]} samples ({train_size/n_samples*100:.1f}%)")
    print(f"Test set: {X_test.shape[0]} samples ({(n_samples-train_size)/n_samples*100:.1f}%)")
    
    # Get corresponding weeks for test set
    test_weeks = pandas_df['year_week'].iloc[train_size + sequence_length:].values
    print(f"Test period: {test_weeks[0]} to {test_weeks[-1]}")
    
    # Build and train LSTM model
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    if problem_type == 'classification':
        if not np.issubdtype(y_train.dtype, np.integer):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_train = le.fit_transform(y_train)
            y_test = le.transform(y_test)
        output_dim = len(np.unique(y_train))
        print(f"\nClassification problem with {output_dim} classes")
    else:
        output_dim = 1
        print("\nRegression problem")
    
    lstm_model = build_lstm_model(input_shape, output_dim, problem_type)
    
    # Train the model on training set only
    print("\nTraining LSTM model on first 70% of data...")
    history = lstm_model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=16,
        validation_split=0.2,  # Use 20% of training data for validation
        verbose=1
    )
    
    # Evaluate on test set (last 30%)
    print("\nEvaluating on test set (last 30% of data)...")
    test_predictions = lstm_model.predict(X_test, verbose=0)
    
    if problem_type == 'regression':
        test_mse = mean_squared_error(y_test, test_predictions)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_rmse = np.sqrt(test_mse)
        
        print(f"\nTest Set Performance:")
        print(f"MSE: {test_mse:.4f}")
        print(f"RMSE: {test_rmse:.4f}")
        print(f"MAE: {test_mae:.4f}")
        
        # Plot predictions vs actual for test set
        plt.figure(figsize=(14, 6))
        plt.plot(range(len(y_test)), y_test, label='Actual', marker='o', alpha=0.7)
        plt.plot(range(len(y_test)), test_predictions, label='Predicted', marker='x', alpha=0.7)
        plt.xlabel('Test Sample Index')
        plt.ylabel(target_column)
        plt.title(f'Test Set: Actual vs Predicted\n{specific_product} (Last 30% of data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    else:
        test_predictions_class = np.argmax(test_predictions, axis=1)
        test_accuracy = accuracy_score(y_test, test_predictions_class)
        print(f"\nTest Set Accuracy: {test_accuracy:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Model Loss During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if problem_type == 'regression':
        plt.subplot(1, 2, 2)
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.title('Model MAE During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy During Training')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Create a wrapper model for SHAP compatibility
    class LSTMWrapper:
        def __init__(self, model, sequence_length, numeric_cols):
            self.model = model
            self.sequence_length = sequence_length
            self.numeric_cols = numeric_cols
            self.n_features = len(numeric_cols)
        
        def predict(self, X):
            if len(X.shape) == 2:
                X_reshaped = X.reshape(-1, self.sequence_length, self.n_features)
                return self.model.predict(X_reshaped, verbose=0)
            return self.model.predict(X, verbose=0)
    
    # Create wrapper instance
    wrapped_model = LSTMWrapper(lstm_model, sequence_length, numeric_cols)
    
    # Use TEST SET for SHAP analysis (last 30%)
    n_samples = min(50, len(X_test))
    X_sample_flattened = X_test[:n_samples].reshape(n_samples, -1)
    
    print(f"\nSHAP analysis on test set samples: {n_samples}")
    print(f"SHAP input shape: {X_sample_flattened.shape}")
    
    # Use a smaller background dataset from test set
    background_size = min(20, len(X_test))
    background_data = X_test[:background_size].reshape(background_size, -1)
    
    print(f"Background data shape: {background_data.shape}")
    
    # Create SHAP explainer
    try:
        print("\nComputing SHAP values...")
        explainer = shap.KernelExplainer(wrapped_model.predict, background_data)
        shap_values = explainer.shap_values(X_sample_flattened)
        
        print(f"SHAP values computed successfully")
        
    except Exception as e:
        print(f"Error with KernelExplainer: {e}")
        print("Trying alternative approach with GradientExplainer...")
        
        def model_output(X):
            return lstm_model(X)
        
        explainer = shap.GradientExplainer(model_output, X_test[:background_size])
        shap_values = explainer.shap_values(X_sample_flattened.reshape(n_samples, sequence_length, len(numeric_cols)))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
    
    # Handle different SHAP values formats
    if isinstance(shap_values, list):
        shap_values_2d = shap_values[0] if len(shap_values) > 0 else shap_values
    else:
        shap_values_2d = shap_values
    
    # Ensure shap_values_2d is 2D
    if len(shap_values_2d.shape) == 3:
        shap_values_2d = shap_values_2d.reshape(shap_values_2d.shape[0], -1)
    
    print(f"Final SHAP values shape: {shap_values_2d.shape}")
    
    # Reshape feature names for sequences
    sequence_feature_names = []
    for i in range(sequence_length):
        for col in numeric_cols:
            sequence_feature_names.append(f"{col}_t-{sequence_length-i-1}")
    
    print(f"Number of feature names: {len(sequence_feature_names)}")
    
    # 1. SHAP Summary Plot
    try:
        plt.figure(figsize=(15, 10))
        shap.summary_plot(shap_values_2d, X_sample_flattened, 
                         feature_names=sequence_feature_names, 
                         show=False)
        plt.title(f"SHAP Summary Plot - {specific_product}\n{target_column} Prediction (Test Set)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in summary plot: {e}")
        print("Creating alternative summary plot...")
        
        mean_shap = np.mean(np.abs(shap_values_2d), axis=0)
        top_indices = np.argsort(mean_shap)[-20:]
        valid_indices = [i for i in top_indices if i < len(sequence_feature_names)]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(valid_indices)), mean_shap[valid_indices])
        plt.yticks(range(len(valid_indices)), [sequence_feature_names[i] for i in valid_indices])
        plt.title(f"Top SHAP Features - {specific_product} (Test Set)")
        plt.xlabel("Mean |SHAP value|")
        plt.tight_layout()
        plt.show()
    
    # 2. SHAP Feature Importance (Bar Plot)
    try:
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values_2d, X_sample_flattened, 
                         feature_names=sequence_feature_names, 
                         plot_type="bar", 
                         show=False)
        plt.title(f"SHAP Feature Importance - {specific_product} (Test Set)")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in bar plot: {e}")
    
    # 3. Aggregate SHAP values by original feature
    feature_importance_agg = {}
    for i, feature_name in enumerate(sequence_feature_names):
        original_feature = feature_name.split('_t-')[0]
        if original_feature not in feature_importance_agg:
            feature_importance_agg[original_feature] = []
        feature_importance_agg[original_feature].append(np.mean(np.abs(shap_values_2d[:, i])))
    
    feature_importance_final = {k: np.mean(v) for k, v in feature_importance_agg.items()}
    
    # Plot aggregated feature importance
    plt.figure(figsize=(10, 6))
    features_sorted = sorted(feature_importance_final.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*features_sorted)
    
    plt.bar(range(len(features)), importance)
    plt.xticks(range(len(features)), features, rotation=45, ha='right')
    plt.title(f"Aggregated SHAP Feature Importance\n{specific_product} (Weekly Averages, Test Set)")
    plt.xlabel("Features")
    plt.ylabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.show()
    
    # 4. Summary statistics
    print(f"\nTop 10 Most Important Features for {specific_product}:")
    shap_summary = pd.DataFrame({
        'Feature': sequence_feature_names,
        'Mean_Abs_SHAP': np.mean(np.abs(shap_values_2d), axis=0)
    })
    top_features = shap_summary.sort_values('Mean_Abs_SHAP', ascending=False).head(10)
    print(top_features)
    
    # 5. Temporal importance analysis
    temporal_importance = {}
    for feature in numeric_cols:
        feature_indices = [i for i, name in enumerate(sequence_feature_names) if name.startswith(feature)]
        if feature_indices:
            temporal_importance[feature] = np.mean(np.abs(shap_values_2d[:, feature_indices]), axis=0)
    
    # Plot temporal importance
    top_original_features = sorted(feature_importance_final.items(), key=lambda x: x[1], reverse=True)[:5]
    
    plt.figure(figsize=(12, 6))
    for feature, _ in top_original_features:
        if feature in temporal_importance:
            plt.plot(range(sequence_length), temporal_importance[feature][::-1], 
                    marker='o', label=f"{feature}", linewidth=2)
    
    plt.xlabel("Time Steps (t-0 = most recent week)")
    plt.ylabel("Mean |SHAP value|")
    plt.title(f"Temporal Feature Importance\n{specific_product} (Weekly Data)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 6. Save results
    results_df = pd.DataFrame({
        'ProductDesignation': [specific_product] * len(features_sorted),
        'Feature': [f[0] for f in features_sorted],
        'SHAP_Importance': [f[1] for f in features_sorted],
        'Test_Period': [f"{test_weeks[0]} to {test_weeks[-1]}"] * len(features_sorted)
    })
    
    results_df.to_csv(f'shap_analysis_{specific_product.replace(" ", "_")}_weekly.csv', index=False)
    print(f"\nResults saved to 'shap_analysis_{specific_product.replace(' ', '_')}_weekly.csv'")
    
    # Save test predictions
    test_results_df = pd.DataFrame({
        'week': test_weeks[:len(y_test)],
        'actual': y_test.flatten() if len(y_test.shape) > 1 else y_test,
        'predicted': test_predictions.flatten() if len(test_predictions.shape) > 1 else test_predictions
    })
    test_results_df.to_csv(f'test_predictions_{specific_product.replace(" ", "_")}_weekly.csv', index=False)
    print(f"Test predictions saved to 'test_predictions_{specific_product.replace(' ', '_')}_weekly.csv'")
    
    print(f"\nLSTM-SHAP analysis for '{specific_product}' completed successfully!")
    print(f"Model trained on first 70% of weekly data, tested on last 30%")

else:
    print(f"Not enough data for {specific_product}. Need more than {sequence_length} weeks after aggregation.")
