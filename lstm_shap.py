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
specific_product = "your_specific_product"  # Change this to your actual product designation

df_specific_product = df.filter(df["ProductDesignation"] == specific_product)

print(f"Number of records for product '{specific_product}': {df_specific_product.count()}")

# Identify numeric columns and target variable
target_column = "AVAFailure"

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
                and f.name != target_column 
                and f.name != "ProductDesignation"
                and f.name != "part_id"
                and not f.name.startswith("id")]

print(f"Numeric features selected: {numeric_cols}")

# Convert to pandas for time series processing
pandas_df = df_specific_product.select(numeric_cols + [target_column, "ProductDesignation"]).toPandas()

if len(pandas_df) == 0:
    raise ValueError(f"No data found for product designation: {specific_product}")
    
print(f"Data shape for {specific_product}: {pandas_df.shape}")

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

# Create sequences
if len(scaled_df) > sequence_length:
    X_sequences, y_sequences = create_sequences(scaled_df, numeric_cols, target_column, sequence_length)
    
    print(f"Sequences created - X shape: {X_sequences.shape}, y shape: {y_sequences.shape}")
    
    # Build and train LSTM model
    input_shape = (X_sequences.shape[1], X_sequences.shape[2])
    
    if problem_type == 'classification':
        if not np.issubdtype(y_sequences.dtype, np.integer):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_sequences = le.fit_transform(y_sequences)
        output_dim = len(np.unique(y_sequences))
        print(f"Classification problem with {output_dim} classes")
    else:
        output_dim = 1
        print("Regression problem")
    
    lstm_model = build_lstm_model(input_shape, output_dim, problem_type)
    
    # Train the model
    print("Training LSTM model...")
    history = lstm_model.fit(
        X_sequences, y_sequences,
        epochs=30,
        batch_size=16,
        validation_split=0.2,
        verbose=1
    )
    
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
    
    # Prepare data for SHAP - use flattened sequences
    n_samples = min(50, len(X_sequences))
    X_sample_flattened = X_sequences[:n_samples].reshape(n_samples, -1)
    
    print(f"SHAP input shape: {X_sample_flattened.shape}")
    
    # Use a smaller background dataset
    background_size = min(20, len(X_sequences))
    background_data = X_sequences[:background_size].reshape(background_size, -1)
    
    print(f"Background data shape: {background_data.shape}")
    
    # Create SHAP explainer
    try:
        explainer = shap.KernelExplainer(wrapped_model.predict, background_data)
        shap_values = explainer.shap_values(X_sample_flattened)
        
        print(f"SHAP values type: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"SHAP values list length: {len(shap_values)}")
            for i, sv in enumerate(shap_values):
                print(f"SHAP values[{i}] shape: {sv.shape}")
        else:
            print(f"SHAP values shape: {shap_values.shape}")
        
    except Exception as e:
        print(f"Error with KernelExplainer: {e}")
        print("Trying alternative approach with GradientExplainer...")
        
        import tensorflow as tf
        def model_output(X):
            return lstm_model(X)
        
        explainer = shap.GradientExplainer(model_output, X_sequences[:background_size])
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
    print(f"X_sample_flattened shape: {X_sample_flattened.shape}")
    
    # Reshape feature names for sequences
    sequence_feature_names = []
    for i in range(sequence_length):
        for col in numeric_cols:
            sequence_feature_names.append(f"{col}_t-{sequence_length-i-1}")
    
    print(f"Number of feature names: {len(sequence_feature_names)}")
    
    # 1. SHAP Summary Plot - with proper shape checking
    try:
        plt.figure(figsize=(15, 10))
        shap.summary_plot(shap_values_2d, X_sample_flattened, 
                         feature_names=sequence_feature_names, 
                         show=False)
        plt.title(f"SHAP Summary Plot - {specific_product}\nAVAFailure Prediction")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in summary plot: {e}")
        print("Creating alternative summary plot...")
        
        # Alternative: Plot mean absolute SHAP values
        mean_shap = np.mean(np.abs(shap_values_2d), axis=0)
        top_indices = np.argsort(mean_shap)[-20:]  # Top 20 features
        
        # FIX: Ensure we don't exceed the feature names length
        valid_indices = [i for i in top_indices if i < len(sequence_feature_names)]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(valid_indices)), mean_shap[valid_indices])
        plt.yticks(range(len(valid_indices)), [sequence_feature_names[i] for i in valid_indices])
        plt.title(f"Top SHAP Features - {specific_product}")
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
        plt.title(f"SHAP Feature Importance - {specific_product}")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in bar plot: {e}")
    
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
    top_features_idx = np.argsort(mean_abs_shap)[-2:]
    
    for i, feature_idx in enumerate(top_features_idx):
        feature_name = sequence_feature_names[feature_idx]
        try:
            plt.figure(figsize=(10, 6))
            feature_values = X_sample_flattened[:, feature_idx]
            shap_values_feature = shap_values_2d[:, feature_idx]
            
            plt.scatter(feature_values, shap_values_feature, alpha=0.6)
            plt.xlabel(feature_name)
            plt.ylabel("SHAP Value")
            plt.title(f"SHAP Dependence: {feature_name}\n{specific_product}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Error in dependence plot for {feature_name}: {e}")
    
    # 5. Summary statistics
    print(f"\nTop 10 Most Important Features for {specific_product}:")
    shap_summary = pd.DataFrame({
        'Feature': sequence_feature_names,
        'Mean_Abs_SHAP': np.mean(np.abs(shap_values_2d), axis=0)
    })
    top_features = shap_summary.sort_values('Mean_Abs_SHAP', ascending=False).head(10)
    print(top_features)
    
    # 6. Temporal importance analysis
    temporal_importance = {}
    for feature in numeric_cols:
        feature_indices = [i for i, name in enumerate(sequence_feature_names) if name.startswith(feature)]
        if feature_indices:
            temporal_importance[feature] = np.mean(np.abs(shap_values_2d[:, feature_indices]), axis=0)
    
    # Plot temporal importance
    top_original_features = sorted(feature_importance_final.items(), key=lambda x: x[1], reverse=True)[:5]
    
    plt.figure(figsize=(12, 6))
    for feature, importance in top_original_features:
        if feature in temporal_importance:
            plt.plot(range(sequence_length), temporal_importance[feature][::-1], 
                    marker='o', label=f"{feature}", linewidth=2)
    
    plt.xlabel("Time Steps (t-0 = most recent)")
    plt.ylabel("Mean |SHAP value|")
    plt.title(f"Temporal Feature Importance\n{specific_product}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # 7. Save results
    results_df = pd.DataFrame({
        'ProductDesignation': [specific_product] * len(features_sorted),
        'Feature': [f[0] for f in features_sorted],
        'SHAP_Importance': [f[1] for f in features_sorted]
    })
    
    results_df.to_csv(f'shap_analysis_{specific_product.replace(" ", "_")}.csv', index=False)
    print(f"\nResults saved to 'shap_analysis_{specific_product.replace(' ', '_')}.csv'")
    print(f"\nLSTM-SHAP analysis for '{specific_product}' completed successfully!")

else:
    print(f"Not enough data for {specific_product}. Need more than {sequence_length} records.")
