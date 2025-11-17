from pyspark.sql import functions as F
from pyspark.sql.window import Window
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType

# Set all random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Your existing code for random part selection WITH fixed seed
random_parts = (
    df.select("part_id")
      .distinct()
      .withColumn("rand", F.rand(seed=RANDOM_SEED))  # Fixed seed for Spark random
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

# Convert to pandas for SHAP analysis
pandas_df = df_random_100_parts.select(numeric_cols + [target_column, "part_id"]).toPandas()

# Separate features and target
X = pandas_df[numeric_cols]
y = pandas_df[target_column]

# Train a model for SHAP analysis with fixed seeds
# For regression:
# model = RandomForestRegressor(
#     n_estimators=100, 
#     random_state=RANDOM_SEED,
#     max_depth=10,  # Added for more stability
#     min_samples_split=10,  # Added for more stability
#     bootstrap=True
# )

# For classification:
model = RandomForestClassifier(
    n_estimators=100, 
    random_state=RANDOM_SEED,
    max_depth=10,  # Added for more stability
    min_samples_split=10,  # Added for more stability
    bootstrap=True
)

model.fit(X, y)

# Calculate SHAP values with fixed random state for TreeExplainer
# Note: TreeExplainer doesn't have a random_state parameter, but we can set numpy seed
np.random.seed(RANDOM_SEED)
explainer = shap.TreeExplainer(
    model, 
    feature_perturbation="interventional"  # More stable than "tree_path_dependent"
)

# Calculate SHAP values
shap_values = explainer.shap_values(X)

# For classification, specify which class if needed
# If it's binary classification, shap_values will have shape [n_samples, n_features]
# If it's multi-class, shap_values will be a list of arrays for each class
if isinstance(shap_values, list):
    # For multi-class, use the first class or choose based on your needs
    shap_values_to_use = shap_values[0]
    expected_value = explainer.expected_value[0]
else:
    shap_values_to_use = shap_values
    expected_value = explainer.expected_value

# 1. SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values_to_use, X, feature_names=numeric_cols, show=False)
plt.title("SHAP Summary Plot - Feature Impact on Model Output")
plt.tight_layout()
plt.show()

# 2. SHAP Feature Importance (Bar Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_to_use, X, feature_names=numeric_cols, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

# 3. SHAP Values DataFrame
shap_df = pd.DataFrame(shap_values_to_use, columns=[f"SHAP_{col}" for col in numeric_cols])
pandas_df_with_shap = pd.concat([pandas_df.reset_index(drop=True), shap_df], axis=1)

# Display first few rows with SHAP values
print("Data with SHAP values (first 10 rows):")
print(pandas_df_with_shap[['part_id', target_column] + list(shap_df.columns[:3])].head(10))

# 4. SHAP Force Plot for first observation
plt.figure(figsize=(12, 4))
shap.force_plot(expected_value, 
                shap_values_to_use[0], 
                X.iloc[0], 
                feature_names=numeric_cols, 
                matplotlib=True, 
                show=False)
plt.title("SHAP Force Plot - First Observation")
plt.tight_layout()
plt.show()

# 5. SHAP Dependence Plot for top features (most stable ones)
# Get mean absolute SHAP values to find most important features
mean_abs_shap = np.abs(shap_values_to_use).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-3:]  # Top 3 features
top_feature_names = [numeric_cols[i] for i in top_features_idx]

print(f"\nTop 3 most important features by mean |SHAP|:")
for i, feature_name in enumerate(top_feature_names):
    print(f"{i+1}. {feature_name}: {mean_abs_shap[top_features_idx[i]]:.4f}")

for feature_name in top_feature_names:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_name, shap_values_to_use, X, feature_names=numeric_cols, show=False)
    plt.title(f"SHAP Dependence Plot for {feature_name}")
    plt.tight_layout()
    plt.show()

# 6. Summary statistics of SHAP values
print("\nSHAP Values Summary Statistics (sorted by importance):")
shap_summary = pd.DataFrame({
    'Feature': numeric_cols,
    'Mean_SHAP': np.mean(shap_values_to_use, axis=0),
    'Std_SHAP': np.std(shap_values_to_use, axis=0),
    'Mean_Abs_SHAP': np.mean(np.abs(shap_values_to_use), axis=0)
})
shap_summary = shap_summary.sort_values('Mean_Abs_SHAP', ascending=False)
print(shap_summary.head(10))  # Show only top 10

# 7. Save the part_ids used for this analysis to ensure reproducibility
print(f"\nPart IDs used in this analysis (first 10): {part_ids[:10]}")
print(f"Total parts analyzed: {len(part_ids)}")

# 8. Save SHAP values to CSV with timestamp for tracking
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'shap_values_analysis_seed{RANDOM_SEED}_{timestamp}.csv'
pandas_df_with_shap.to_csv(filename, index=False)
print(f"\nSHAP values saved to '{filename}'")

# 9. Additional stability: Verify model performance is consistent
from sklearn.metrics import accuracy_score, classification_report
y_pred = model.predict(X)
if hasattr(model, 'predict_proba'):
    y_proba = model.predict_proba(X)
    print(f"\nModel accuracy: {accuracy_score(y, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
