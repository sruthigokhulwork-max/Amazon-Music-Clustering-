# Step 1: importing necessary libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ast

os.makedirs("outputs", exist_ok=True)

# step 2: loading the data
df = pd.read_csv("outputs/clustered_songs.csv")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Cluster counts:")
print(df['cluster'].value_counts().sort_index())

# Step 3: defining the audio features and cluster names

audio_features = [
    'danceability', 'energy', 'loudness', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness',
    'valence', 'tempo', 'duration_ms'
]

cluster_names = {
    0: 'Calm Acoustic Songs',
    1: 'High Energy Mainstream',
    2: 'Speech Heavy & Live',
    3: 'Pure Instrumental'
}
# here we use the 10 audio features that are useful for human interpertation.

# Step 4: cluster profiling

# Calculating mean of every audio feature per cluster
cluster_profile = df.groupby('cluster')[audio_features].mean()

print("CLUSTER PROFILES — Mean values per cluster")
print(cluster_profile.round(3).to_string())

# Step 5: feature averages per cluster(BAR CHART)
# we create the bar chart 
# this helps show how our features differ.

fig, axes = plt.subplots(2, 5, figsize=(22, 10))
axes = axes.flatten()

for i, feature in enumerate(audio_features):
    values = cluster_profile[feature]
    bars = axes[i].bar(
        [f'C{c}' for c in range(4)],
        values,
        color=['#e41a1c', '#377eb8', '#4daf4a', '#984ea3'],
        edgecolor='white',
        alpha=0.85
    )
    axes[i].set_title(feature, fontsize=11, fontweight='bold')
    axes[i].set_ylabel("Mean Value")
    axes[i].set_xlabel("Cluster")

    # adding value labels on top of each bar
    for bar, val in zip(bars, values):
        axes[i].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{val:.2f}',
            ha='center', va='bottom', fontsize=8
        )

plt.suptitle("Average Feature Values per Cluster",
             fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig("outputs/cluster_feature_bars.png", dpi=150, bbox_inches='tight')
plt.show()

# Step 6: Feature averages per cluster(HEATMAP)
# here we normalizing the profiles since loudness is in dB and danceability is 0-1.

# Normalize the profile for fair comparison across features
# (since loudness is in dB and danceability is 0-1)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
profile_normalized = pd.DataFrame(
    scaler.fit_transform(cluster_profile),
    index=cluster_profile.index,
    columns=cluster_profile.columns
)

plt.figure(figsize=(14, 5))
sns.heatmap(
    profile_normalized,
    annot=cluster_profile.round(2),
    fmt='',
    cmap='RdYlGn',
    linewidths=0.5,
    cbar_kws={'label': 'Normalized Value (0=lowest, 1=highest)'}
)
plt.title("Cluster Feature Heatmap\n(Color = relative level, Number = actual mean)",
          fontsize=13, fontweight='bold')
plt.xlabel("Audio Feature")
plt.ylabel("Cluster")
plt.tight_layout()
plt.savefig("outputs/cluster_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# Step 7: validating the genre

print("GENRE VALIDATION — Top genres per cluster")
# genres column is stored as string like "['pop', 'rock']"
# We need to parse it back into a real list
def parse_genres(genre_str):
    try:
        result = ast.literal_eval(genre_str)
        if isinstance(result, list) and len(result) > 0:
            return result[0]  # take first genre
        return 'unknown'
    except:
        return 'unknown'

df['primary_genre'] = df['genres'].apply(parse_genres)

for cluster_id in range(4):
    cluster_songs = df[df['cluster'] == cluster_id]
    top_genres = (cluster_songs['primary_genre']
                  .value_counts()
                  .head(10))
    print(f"\nCluster {cluster_id} — Top 10 Genres:")
    for genre, count in top_genres.items():
        pct = count / len(cluster_songs) * 100
        print(f"   {genre:<30} {count:>5} songs ({pct:.1f}%)")

# Step 8: top tracks per cluster

print("TOP 5 SONGS PER CLUSTER")

for cluster_id in range(4):
    cluster_songs = df[df['cluster'] == cluster_id]
    top_songs = cluster_songs.nlargest(5, 'popularity_songs')[
        ['name_song', 'name_artists', 'popularity_songs']
    ]
    print(f"\nCluster {cluster_id} — {cluster_names[cluster_id]}")
    for _, row in top_songs.iterrows():
        print(f"   {row['name_song']:<40} by {row['name_artists']:<25} popularity: {row['popularity_songs']}")

# Step 9: summary

cluster_names = {
    0: 'Calm Acoustic Songs',
    1: 'High Energy Mainstream',
    2: 'Speech Heavy & Live',
    3: 'Pure Instrumental'
}

print("final cluster summary")


for cluster_id in range(4):
    cluster_songs = df[df['cluster'] == cluster_id]
    profile = cluster_profile.loc[cluster_id]
    print(f"""
Cluster {cluster_id} — {cluster_names[cluster_id]}
  Songs      : {len(cluster_songs):,} ({len(cluster_songs)/len(df)*100:.1f}%)
  Danceability : {profile['danceability']:.3f}
  Energy       : {profile['energy']:.3f}
  Loudness     : {profile['loudness']:.3f} dB
  Acousticness : {profile['acousticness']:.3f}
  Speechiness  : {profile['speechiness']:.3f}
  Valence      : {profile['valence']:.3f}
  Tempo        : {profile['tempo']:.1f} BPM
    """)

# Step 10: saving the final csv.

df['cluster_name'] = df['cluster'].map(cluster_names)

df.to_csv("outputs/final_clustered_songs.csv", index=False)
print("Saved: outputs/final_clustered_songs.csv")
print(f"Total songs: {len(df):,}")
print("Columns:", df.columns.tolist())