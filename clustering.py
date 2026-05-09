# Step 1: importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN
import os


os.makedirs("outputs", exist_ok=True)


# Step 2: loading the required datas

x_scaled = pd.read_csv("outputs/x_scaled.csv").values
# here we convert the dataframe into a numpy array since the k-means works on numpy array.
pca_df = pd.read_csv("outputs/x_pca_2d.csv")
df_ref = pd.read_csv("outputs/drop_column.csv")

print("shape of x_scaled")
print(x_scaled.shape)
print("shape of x_pca_2d")
print(pca_df.shape)

# Step 3: K-means clustering
# it means that an algorithm that groups data points into K clusters by minimizing the distance between points 
# and their group's center.


# Step 3.1: elbow method
# it is a visual technique used to determine the optimal number of clusters by finding the point 
# where adding more groups yields diminishing returns.

inertia_values = []
k_range = range(2, 16)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(x_scaled)
    inertia_values.append(kmeans.inertia_)
    print(f"k={k} | Inertia: {kmeans.inertia_:.0f}")

plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, marker='o', color='steelblue', linewidth=2, markersize=8)
plt.xlabel("Number of Clusters (k)", fontsize=12)
plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=12)
plt.title("Elbow Method — Finding the Best K", fontsize=14, fontweight='bold')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("outputs/elbow_plot.png", dpi=150, bbox_inches='tight')
plt.show()

# inertia measures how compact our clusters are.
# inertia decreases as k increases.

# from the output we can see our graph is a gradual curve with mo sharp elbow.
# but from k=2 to k=5 we can see our curve drops steeply and then from k=5 till k=15 drops gradually.
# so the bend happens from k=5.


# Step 3.2: Silhouette Score
# its a  metric ranging from -1 to 1 that evaluates how well-separated and cohesive our clusters are.

silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels, sample_size=10000, random_state=42)
    silhouette_scores.append(score)
    print(f"k={k} | Silhouette Score: {score:.4f}")

plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o', color='green', linewidth=2, markersize=8)
plt.xlabel("Number of Clusters (K)", fontsize=12)
plt.ylabel("Silhouette Score", fontsize=12)
plt.title("Silhouette Score — Higher is Better", fontsize=14, fontweight='bold')
plt.xticks(k_range)
plt.tight_layout()
plt.savefig("outputs/silhouette_plot.png", dpi=150, bbox_inches='tight')
plt.show()

# The best K is where silhouette score is highest.
# from this we understand that highest score is at K=3, drops sharply at K=4 and K=5, 
# then becomes flat and low from K=6 onwards.

# even though k=3 has the highest score, having only 3 clusters for our dataset makes it too broad, its not useful.
# k=4 - 2nd best and is nearer to the elbow, having extra cluster is useful

# from this, we found that k = 4 is the best K 


# Step 4: applying the best K

# After looking at your elbow and silhouette plots, choose K
# We'll use K=6 as a starting point — adjust based on your plots
best_k = 4

kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(x_scaled)

print("Cluster label sample:", cluster_labels[:10])
print("Unique clusters:", np.unique(cluster_labels))

# Count how many songs are in each cluster
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
print("\nSongs per cluster:")
for cluster, count in cluster_counts.items():
    print(f"  Cluster {cluster}: {count:,} songs ({count/len(cluster_labels)*100:.1f}%)")


# Step 5: evaluating the results:
final_silhouette = silhouette_score(x_scaled, cluster_labels, sample_size=10000, random_state=42)
final_db = davies_bouldin_score(x_scaled, cluster_labels)

print("FINAL CLUSTER EVALUATION")
print(f"Silhouette Score   : {final_silhouette:.4f}  (higher is better, max=1)")
print(f"Davies-Bouldin Index: {final_db:.4f}  (lower is better, min=0)")
print(f"Inertia            : {kmeans_final.inertia_:.0f}")

# davies-bouldin index measures the average similarity between each cluster and its most similar other cluster. Lower is better. 
# 0 means perfect separation. Typical good values are below 1.0.


# Step 6: colouring the role by cluster:

colors = ['#e41a1c', '#377eb8', '#4daf4a','#984ea3',
          '#ff7f00', '#a65628', '#f781bf','#999999']

pca_df['cluster'] = cluster_labels

plt.figure(figsize=(12, 8))
for cluster in range(best_k):
    mask = pca_df['cluster'] == cluster
    plt.scatter(
        pca_df.loc[mask, 'PC1'],
        pca_df.loc[mask, 'PC2'],
        c=colors[cluster],
        alpha=0.3,
        s=1,
        label=f'Cluster {cluster} ({mask.sum():,} songs)'
    )

plt.xlabel("PC1 — Loudness axis", fontsize=12)
plt.ylabel("PC2 — Danceability axis", fontsize=12)
plt.title(f"2D PCA — {best_k} Clusters of Songs", fontsize=14, fontweight='bold')
plt.legend(markerscale=8, fontsize=9)
plt.tight_layout()
plt.savefig("outputs/pca_clusters_colored.png", dpi=150, bbox_inches='tight')
plt.show()


# Step 7: saving the results:
df_cleaned = pd.read_csv("cleaned_single_genre_artists.csv")
df_cleaned['cluster'] = cluster_labels

df_final = pd.concat([df_ref, df_cleaned], axis=1)
df_final.to_csv("outputs/clustered_songs.csv", index=False)

# pd.concat joins 2 dataframes side by side.

# Step 8: DBSCAN — Comparison Algorithm 

print("DBSCAN — COMPARISON")

dbscan = DBSCAN(eps=1.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(x_scaled)

n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"DBSCAN found    : {n_clusters} clusters")
print(f"Noise points    : {n_noise} ({n_noise/len(x_scaled)*100:.1f}%)")
print(f"K-Means K=4 Silhouette : 0.1669")
print("Conclusion: K-Means chosen as final model")
print("Reason: DBSCAN struggles with high-dimensional")
print("uniform density data like audio features.")