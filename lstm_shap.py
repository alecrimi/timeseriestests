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

# Select specific product designation instead of random parts
# Replace 'your_specific_product' with the actual product designation you want to analyze
specific_product = "your_specific_product"  # Change this to your actual product designation

df_specific_product = df.filter(df["Product designation"] == specific_product)

# If you want to analyze multiple specific products, you can use:
# specific_products = ["product1", "product2", "product3"]
# df_specific_product = df.filter(df["Product designation"].isin(specific_products))

print(f"Number of records for product '{specific_product}': {df_specific_product.count()}")

# Identify numeric columns and target variable
target_column = "Failed_quantity"  # Using Failed_quantity as target

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
                and f.name != target_column 
                and f.name != "Product designation"
                and f.name != "part_id"  # Remove part_id if it exists
                and not f.name.startswith("id")]  # Typically exclude ID columns

print(f"Numeric features selected: {numeric_cols}")

# Convert to pandas for time series processing
pandas_df = df_specific_product.select(numeric_cols + [target_column, "Product designation"]).toPandas()

# Check if we have enough data
if len(pandas_df) == 0:
    raise ValueError(f"No data found for product designation: {specific_product}")
    
print(f"Data shape for {specific_product}: {pandas_df.shape}")

# Sort by time if you have a timestamp column
# If you have a timestamp column, add it to the select statement and sort by it
# pandas_df = pandas_df.sort_values('timestamp_column')

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

# Create sequences
if len(scaled_df) > sequence_length:
    X_sequences, y_sequences = create_sequences(scaled_df, numeric_cols, target_column, sequence_length)
    
    print(f"Sequences created: {X_sequences.shape}")
    print(f"Target shape: {y_sequences.shape}")
    
    # Build and train LSTM model
    input_shape = (X_sequences.shape[1], X_sequences.shape[2])
    
    if problem_type == 'classification':
        # For classification, ensure labels are integers
        if not np.issubdtype(y_sequences.dtype, np.integer):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_sequences = le.fit_transform(y_sequences)
        output_dim = len(np.unique(y_sequences))
    else:
        output_dim = 1
    
    lstm_model = build_lstm_model(input_shape, output_dim, problem_type)
    
    # Train the model
    print("Training LSTM model...")
    history = lstm_model.fit(
        X_sequences, y_sequences,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    if problem_type == 'regression':
        plt.plot(history.history['mae'], label='Training MAE')
        plt.plot(history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.ylabel('MAE')
    else:
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
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
    background_data = X_sequences[:50].reshape(50, -1)
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(wrapped_model.predict, background_data)
    
    # Calculate SHAP values for a subset of data
    sample_size = min(50, len(X_sequences))
    sample_indices = np.random.choice(len(X_sequences), sample_size, replace=False)
    X_sample = X_sequences[sample_indices].reshape(sample_size, -1)
    
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
        expected_value = explainer.expected_value[0]
    else:
        shap_values_2d = shap_values
        expected_value = explainer.expected_value
    
    # 1. SHAP Summary Plot
    plt.figure(figsize=(15, 10))
    shap.summary_plot(shap_values_2d, X_sample, feature_names=sequence_feature_names, show=False)
    plt.title(f"SHAP Summary Plot - {specific_product}\nFailed_quantity Prediction")
    plt.tight_layout()
    plt.show()
    
    # 2. SHAP Feature Importance (Bar Plot)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values_2d, X_sample, feature_names=sequence_feature_names, 
                     plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {specific_product}")
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
    plt.title(f"Aggregated SHAP Feature Importance\n{specific_product}")
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
        plt.title(f"SHAP Dependence Plot for {feature_name}\n{specific_product}")
        plt.tight_layout()
        plt.show()
    
    # 5. Summary statistics of SHAP values
    print(f"\nSHAP Values Summary Statistics for {specific_product}:")
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
    for feature, importance in top_original_features:
        if feature in temporal_importance:
            plt.plot(range(sequence_length), temporal_importance[feature][::-1], 
                    marker='o', label=f"{feature} (imp: {importance:.4f})", linewidth=2)
    
    plt.xlabel("Time Steps (t-0 = most recent)")
    plt.ylabel("Mean |SHAP value|")
    plt.title(f"Temporal Feature Importance\n{specific_product} - Failed_quantity Prediction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 7. Save results
    results_df = pd.DataFrame({
        'Product_Designation': [specific_product] * len(features_sorted),
        'Feature': [f[0] for f in features_sorted],
        'SHAP_Importance': [f[1] for f in features_sorted]
    })
    
    results_df.to_csv(f'shap_analysis_{specific_product.replace(" ", "_")}.csv', index=False)
    print(f"\nResults saved to 'shap_analysis_{specific_product.replace(' ', '_')}.csv'")
    print(f"\nLSTM-SHAP analysis for '{specific_product}' completed successfully!")

else:
    print(f"Not enough data for product '{specific_product}'. Need more than {sequence_length} records. Current records: {len(scaled_df)}")
