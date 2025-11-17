from pyspark.sql import functions as F
from pyspark.sql.window import Window
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType

# TensorFlow/Keras for LSTM
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

# Your existing code for random part selection
random_parts = (
    df.select("part_id")
      .distinct()
      .withColumn("rand", F.rand())
      .withColumn("rank", F.row_number().over(Window.orderBy("rand")))
      .filter(F.col("rank") <= 100)
)

part_ids = [row["part_id"] for row in random_parts.collect()]
df_random_100_parts = df.filter(df.part_id.isin(part_ids))

# Identify numeric columns and target variable
target_column = "your_target_column"  # Change this to your actual target column

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
                and f.name != target_column and f.name != "part_id"]

# Convert to pandas for time series processing
pandas_df = df_random_100_parts.select(numeric_cols + [target_column, "part_id"]).toPandas()

# Sort by part_id and ensure temporal order if you have a timestamp column
# If you have a timestamp, add it to the select statement above and sort by it
# pandas_df = pandas_df.sort_values(['part_id', 'timestamp'])

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
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
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
sequence_length = 10  # Adjust based on your time series characteristics

# Group by part_id and create sequences
all_sequences_X = []
all_sequences_y = []

for part_id in pandas_df['part_id'].unique():
    part_data = pandas_df[pandas_df['part_id'] == part_id].reset_index(drop=True)
    
    if len(part_data) > sequence_length:
        # Scale features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(part_data[numeric_cols])
        scaled_df = pd.DataFrame(scaled_features, columns=numeric_cols)
        scaled_df[target_column] = part_data[target_column].values
        
        X_seq, y_seq = create_sequences(scaled_df, numeric_cols, target_column, sequence_length)
        all_sequences_X.append(X_seq)
        all_sequences_y.append(y_seq)

# Combine all sequences
if all_sequences_X:
    X_combined = np.vstack(all_sequences_X)
    y_combined = np.concatenate(all_sequences_y)
    
    # Determine problem type
    unique_targets = np.unique(y_combined)
    problem_type = 'classification' if len(unique_targets) <= 10 else 'regression'
    
    if problem_type == 'classification':
        # For classification, convert to integer labels if needed
        if not np.issubdtype(y_combined.dtype, np.integer):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_combined = le.fit_transform(y_combined)
        output_dim = len(np.unique(y_combined))
    else:
        output_dim = 1
    
    # Build and train LSTM model
    input_shape = (X_combined.shape[1], X_combined.shape[2])
    lstm_model = build_lstm_model(input_shape, output_dim, problem_type)
    
    # Train the model
    history = lstm_model.fit(
        X_combined, y_combined,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Create a wrapper model for SHAP compatibility
    class LSTMWrapper:
        def __init__(self, model, sequence_length, numeric_cols):
            self.model = model
            self.sequence_length = sequence_length
            self.numeric_cols = numeric_cols
        
        def predict(self, X):
            # Reshape for LSTM if needed
            if len(X.shape) == 2:
                X = X.reshape(-1, self.sequence_length, len(self.numeric_cols))
            return self.model.predict(X, verbose=0)
    
    # Create wrapper instance
    wrapped_model = LSTMWrapper(lstm_model, sequence_length, numeric_cols)
    
    # Prepare background data for SHAP (use a subset for efficiency)
    background_data = X_combined[:100].reshape(100, -1)
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(wrapped_model.predict, background_data)
    
    # Calculate SHAP values for a subset of data
    sample_indices = np.random.choice(len(X_combined), min(100, len(X_combined)), replace=False)
    X_sample = X_combined[sample_indices].reshape(len(sample_indices), -1)
    
    shap_values = explainer.shap_values(X_sample)
    
    # Reshape feature names for sequences
    sequence_feature_names = []
    for i in range(sequence_length):
        for col in numeric_cols:
            sequence_feature_names.append(f"{col}_t-{sequence_length-i-1}")
    
    # Convert to numpy array for easier handling
    shap_values = np.array(shap_values)
    
    # Handle different output shapes for regression vs classification
    if len(shap_values.shape) > 2:
        # For multi-class classification, explain the first class
        shap_values_2d = shap_values[0]
    else:
        shap_values_2d = shap_values
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(15, 10))
    shap.summary_plot(shap_values_2d, X_sample, feature_names=sequence_feature_names, show=False)
    plt.title("SHAP Summary Plot - LSTM Feature Impact on Model Output")
    plt.tight_layout()
    plt.show()
    
    # 2. SHAP Feature Importance (Bar Plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_2d, X_sample, feature_names=sequence_feature_names, 
                     plot_type="bar", show=False)
    plt.title("SHAP Feature Importance - LSTM")
    plt.tight_layout()
    plt.show()
    
    # 3. Aggregate SHAP values by original feature (across time steps)
    feature_importance_agg = {}
    for i, feature_name in enumerate(sequence_feature_names):
        original_feature = feature_name.split('_t-')[0]
        if original_feature not in feature_importance_agg:
            feature_importance_agg[original_feature] = []
        feature_importance_agg[original_feature].append(np.mean(np.abs(shap_values_2d[:, i])))
    
    # Calculate mean importance for each original feature
    feature_importance_final = {k: np.mean(v) for k, v in feature_importance_agg.items()}
    
    # Plot aggregated feature importance
    plt.figure(figsize=(10, 6))
    features_sorted = sorted(feature_importance_final.items(), key=lambda x: x[1], reverse=True)
    features, importance = zip(*features_sorted)
    
    plt.bar(range(len(features)), importance)
    plt.xticks(range(len(features)), features, rotation=45)
    plt.title("Aggregated SHAP Feature Importance (Across Time Steps)")
    plt.xlabel("Features")
    plt.ylabel("Mean |SHAP value|")
    plt.tight_layout()
    plt.show()
    
    # 4. SHAP Dependence Plot for top features
    mean_abs_shap = np.abs(shap_values_2d).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:]  # Top 3 sequence features
    top_feature_names = [sequence_feature_names[i] for i in top_features_idx]
    
    for feature_name in top_feature_names[:2]:  # Plot top 2 for clarity
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(feature_name, shap_values_2d, X_sample, 
                           feature_names=sequence_feature_names, show=False)
        plt.title(f"SHAP Dependence Plot for {feature_name}")
        plt.tight_layout()
        plt.show()
    
    # 5. Summary statistics of SHAP values
    print("\nSHAP Values Summary Statistics (LSTM):")
    shap_summary = pd.DataFrame({
        'Sequence_Feature': sequence_feature_names,
        'Mean_SHAP': np.mean(shap_values_2d, axis=0),
        'Std_SHAP': np.std(shap_values_2d, axis=0),
        'Mean_Abs_SHAP': np.mean(np.abs(shap_values_2d), axis=0)
    })
    print(shap_summary.sort_values('Mean_Abs_SHAP', ascending=False).head(10))
    
    # 6. Temporal importance analysis
    temporal_importance = {}
    for feature in numeric_cols:
        feature_indices = [i for i, name in enumerate(sequence_feature_names) if name.startswith(feature)]
        if feature_indices:
            temporal_importance[feature] = np.mean(np.abs(shap_values_2d[:, feature_indices]), axis=0)
    
    # Plot temporal importance for top features
    top_original_features = sorted(feature_importance_final.items(), key=lambda x: x[1], reverse=True)[:5]
    
    plt.figure(figsize=(12, 8))
    for feature, _ in top_original_features:
        if feature in temporal_importance:
            plt.plot(range(sequence_length), temporal_importance[feature][::-1], 
                    marker='o', label=feature, linewidth=2)
    
    plt.xlabel("Time Steps (t-0 = most recent)")
    plt.ylabel("Mean |SHAP value|")
    plt.title("Temporal Feature Importance - How Far Back Features Matter")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\nLSTM-SHAP analysis completed successfully!")

else:
    print("Not enough data to create sequences. Try reducing sequence_length or selecting more parts.")
