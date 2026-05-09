
# Step 1: importing necessary libraries.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

os.makedirs("outputs", exist_ok=True)

# Step 2: data load
df = pd.read_csv("cleaned_single_genre_artists.csv")
df_ref = pd.read_csv("outputs/drop_column.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df_ref.head())

# Step 3: feature selection.
# here we select only the columns having features of how a song sounds.

clustering_features =  ['danceability','energy','loudness','speechiness',
                        'acousticness','instrumentalness','liveness','valence','tempo',
                        'duration_ms','key','mode','time_signature']

# so we use 13 columns and the remaining will be dropped.

dropped_features = [ 'popularity_songs','explicit','release_date','followers', 'popularity_artists']

x = df[clustering_features].copy()
# here we copy the clustering features since we are using them for clustering, if at all changes, we can directly use 
# only those features instead of changing in the raw data which might be a lot of work.

# Step 4: normalization using standard scaler.
# here we are scaling the values to a common range using standard deviations and means.

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled_df = pd.DataFrame(x_scaled, columns = clustering_features)

# printing the scaled to verify.
print("Before scaling - duration_ms range:", x['duration_ms'].min(), "to", x['duration_ms'].max())
print("After scaling  - duration_ms range:", round(x_scaled_df['duration_ms'].min(), 2), 
      "to", round(x_scaled_df['duration_ms'].max(), 2))

print("Before scaling - danceability range:", x['danceability'].min(), "to", x['danceability'].max())
print("After scaling  - danceability range:", round(x_scaled_df['danceability'].min(), 2),
       "to", round(x_scaled_df['danceability'].max(), 2))

# Step 5: visual reprensentation of the scaling

# columns used = danceability, duration_ms, loudness, tempo
# here we compare these four specific features because they represent the most extreme range differences in the dataset.

fig, axes = plt.subplots(2, 4, figsize=(20, 8))

compare_features = ['loudness', 'tempo', 'duration_ms', 'danceability']

for i, col in enumerate(compare_features):
    axes[0, i].hist(x[col], bins=50, color='salmon', edgecolor='white', alpha=0.8)
    axes[0, i].set_title(f'{col} — BEFORE', fontweight='bold')
    axes[0, i].set_xlabel("Original Value")
    axes[0, i].set_ylabel("Count")

    axes[1, i].hist(x_scaled_df[col], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[1, i].set_title(f'{col} — AFTER', fontweight='bold')
    axes[1, i].set_xlabel("Z-score")
    axes[1, i].set_ylabel("Count")

plt.suptitle("Before vs After StandardScaler", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("outputs/scaling_comparison.png", dpi=150, bbox_inches='tight')
plt.show()

# Step 6.1: PCA analysis
# here we transform a large set correlated data into a smaller set of uncorrelated "principal components" to simplify 
# our visualization.
# it reduces our 13d dimensions into a 2d.

pca_full = PCA(n_components=13, random_state=42)
pca_full.fit(x_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("Variance captured per component:")
for i, (ind, cum) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"  PC{i+1}: {ind*100:.1f}%  |  Cumulative: {cum*100:.1f}%")

plt.figure(figsize=(10, 5))
plt.plot(range(1, 14), cumulative_variance * 100, marker='o', color='steelblue', linewidth=2)
plt.axhline(y=80, color='red', linestyle='--', label='80% threshold')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained (%)")
plt.title("PCA — How many components do we need?")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/pca_variance.png", dpi=150, bbox_inches='tight')
plt.show()

# Step 6.2: applying 2d 
# here it reduces the dataset to two principal components to create a 2D visual representation.
# pc1 - loudness
# pc2 = danceability

pca_2d = PCA(n_components=2, random_state=42)
x_pca_2d = pca_2d.fit_transform(x_scaled)

print("Original shape:", x_scaled.shape)
print("After PCA shape:", x_pca_2d.shape)
print("PC1 captures:", round(pca_2d.explained_variance_ratio_[0] * 100, 1), "%")
print("PC2 captures:", round(pca_2d.explained_variance_ratio_[1] * 100, 1), "%")

plt.figure(figsize=(10, 7))
plt.scatter(x_pca_2d[:, 0], x_pca_2d[:, 1],
            alpha=0.1, s=1, color='steelblue')
plt.xlabel("PC1 — Loudness axis")
plt.ylabel("PC2 — Danceability axis")
plt.title("2D PCA of 95,837 Songs")
plt.tight_layout()
plt.savefig("outputs/pca_2.png", dpi=150, bbox_inches='tight')
plt.show()

# Step 7: saving the outputs

x_scaled_df.to_csv("outputs/x_scaled.csv", index=False)

pca_df = pd.DataFrame(x_pca_2d, columns=['PC1', 'PC2'])
pca_df.to_csv("outputs/x_pca_2d.csv", index=False)

pd.Series(clustering_features).to_csv("outputs/clustering_features.csv",
                                       index=False, header=False)


