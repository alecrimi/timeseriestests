import pandas as pd
import numpy as np

# Read your CSV file
df = pd.read_csv("your_file.csv")

# Number of categories (1, 2, 3, and 0 treated as 4th)
num_categories = 4

def encode_category(cat):
    vec = np.zeros(num_categories, dtype=int)
    # map 0 → index 3, else (1→0, 2→1, 3→2)
    index = 3 if cat == 0 else cat - 1
    vec[index] = 1
    return vec.tolist()

# Apply the encoding
df["Category_vector"] = df["Category"].apply(encode_category)

# Optional: drop the original category column
# df = df.drop(columns=["Category"])

# Save or inspect the result
print(df.head())
# df.to_csv("encoded.csv", index=False)
