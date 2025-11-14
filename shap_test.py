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
# You need to specify your target column - replace 'target_column' with actual column name
target_column = "your_target_column"  # Change this to your actual target column

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))
                and f.name != target_column and f.name != "part_id"]

# Convert to pandas for SHAP analysis
pandas_df = df_random_100_parts.select(numeric_cols + [target_column, "part_id"]).toPandas()

# Separate features and target
X = pandas_df[numeric_cols]
y = pandas_df[target_column]

# Train a model for SHAP analysis
# Choose appropriate model based on your problem type

# For regression:
# model = RandomForestRegressor(n_estimators=100, random_state=42)
# model.fit(X, y)

# For classification:
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# For classification, you might want to specify which class to explain
# shap_values = explainer.shap_values(X)[1]  # for class 1

# 1. SHAP Summary Plot
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, feature_names=numeric_cols, show=False)
plt.title("SHAP Summary Plot - Feature Impact on Model Output")
plt.tight_layout()
plt.show()

# 2. SHAP Feature Importance (Bar Plot)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, X, feature_names=numeric_cols, plot_type="bar", show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.show()

# 3. SHAP Values DataFrame
# Create a DataFrame with SHAP values
if len(shap_values.shape) > 2:  # For multi-class classification
    shap_df = pd.DataFrame(shap_values[0], columns=[f"SHAP_{col}" for col in numeric_cols])
else:  # For binary classification or regression
    shap_df = pd.DataFrame(shap_values, columns=[f"SHAP_{col}" for col in numeric_cols])

# Add SHAP values to original dataframe
pandas_df_with_shap = pd.concat([pandas_df.reset_index(drop=True), shap_df], axis=1)

# Display first few rows with SHAP values
print("Data with SHAP values (first 10 rows):")
print(pandas_df_with_shap.head(10))

# 4. SHAP Force Plot for first observation
plt.figure(figsize=(12, 4))
shap.force_plot(explainer.expected_value[0] if hasattr(explainer.expected_value, '__iter__') else explainer.expected_value, 
                shap_values[0], X.iloc[0], feature_names=numeric_cols, matplotlib=True, show=False)
plt.title("SHAP Force Plot - First Observation")
plt.tight_layout()
plt.show()

# 5. SHAP Dependence Plot for top features
# Get mean absolute SHAP values to find most important features
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_features_idx = np.argsort(mean_abs_shap)[-3:]  # Top 3 features
top_feature_names = [numeric_cols[i] for i in top_features_idx]

for feature_name in top_feature_names:
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(feature_name, shap_values, X, feature_names=numeric_cols, show=False)
    plt.title(f"SHAP Dependence Plot for {feature_name}")
    plt.tight_layout()
    plt.show()

# 6. Summary statistics of SHAP values
print("\nSHAP Values Summary Statistics:")
shap_summary = pd.DataFrame({
    'Feature': numeric_cols,
    'Mean_SHAP': np.mean(shap_values, axis=0),
    'Std_SHAP': np.std(shap_values, axis=0),
    'Mean_Abs_SHAP': np.mean(np.abs(shap_values), axis=0)
})
print(shap_summary.sort_values('Mean_Abs_SHAP', ascending=False))

# 7. Save SHAP values to CSV (optional)
pandas_df_with_shap.to_csv('shap_values_analysis.csv', index=False)
print("\nSHAP values saved to 'shap_values_analysis.csv'")
