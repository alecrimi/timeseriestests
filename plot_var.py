from pyspark.sql.types import IntegerType, DoubleType, FloatType, LongType

numeric_cols = [f.name for f in df.schema.fields
                if isinstance(f.dataType, (IntegerType, DoubleType, FloatType, LongType))]


from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(inputCols=numeric_cols, outputCol="features")
df_vec = assembler.transform(df)

from pyspark.ml.feature import PCA

pca = PCA(k=10, inputCol="features", outputCol="pca_features")
pca_model = pca.fit(df_vec)
df_pca = pca_model.transform(df_vec)

print("Explained variance by component:")
print(pca_model.explainedVariance.toArray())

pandas_df = df_pca.select("part_id", "pca_features").toPandas()

# Split vector into separate columns
import pandas as pd
pca_values = pd.DataFrame(pandas_df['pca_features'].tolist(),
                          columns=[f'PC{i+1}' for i in range(10)])
pandas_df = pd.concat([pandas_df['part_id'], pca_values], axis=1)


import matplotlib.pyplot as plt

plt.figure(figsize=(7,6))
for part, subset in pandas_df.groupby('part_id'):
    plt.scatter(subset['PC1'], subset['PC2'], label=part, alpha=0.7)


plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Production Data (Parts)")
plt.legend()
plt.show()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(8,7))
ax = fig.add_subplot(111, projection='3d')

for part, subset in pandas_df.groupby('part_id'):
    ax.scatter(subset['PC1'], subset['PC2'], subset['PC3'], label=part, alpha=0.7)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title("3D PCA of Production Data")
ax.legend()
plt.show()
