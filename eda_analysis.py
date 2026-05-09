
#  AMAZON MUSIC CLUSTERING


# Dataset having songs with audio features with no genre information. 
# The task is to cluster the songs based on their audio features and then analyze the 
# clusters to gain insights.


# Step 1: Importing necessary libraries

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs", exist_ok=True) 
# This line creates a directory "outputs" where we can save our results, such as plots and analysis outputs. The exist_ok=True parameter ensures that if the directory already exists, it won't raise an error. This is useful for organizing our results and keeping our workspace tidy.

# Step 2: Load the dataset

df = pd.read_csv('single_genre_artists.csv')

# Step 3: Exploratory Data Analysis (EDA)

# Step 3.1: Data exploration

print(df.shape)
print(df.head())
print(df.info())
print(df.describe())
print(df.columns.tolist())

# Step 3.2: Checking for missing values

print(" Number of missing values: ")
print(df.isnull().sum())
# From this we found our datset has no missing values.

# Step 3.3: Checking for duplicates 

print("Number of duplicates: ")
print(df.duplicated().sum())
# From this we found our datset has no duplicates.

# Step 3.4: Dropping columns.
#  Since our focus is on clustering the songs based on the audio features which are numerical values,
# we can drop the "str" columns(only 5) from our dataset. They have no relevance. The 6th column 'release_date' is a string 
# but is a date column so we convert it to int which may be used for the clustering. 

# before dropping, lets keep a copy of the columns we will be dropping for references.
 
drop_column = ['id_artists', 'name_artists', 'genres', 'id_songs', 'name_song']
df_drop = df[drop_column].copy()

# Now we can drop the columns from our original dataset.
df.drop(columns=drop_column, inplace = True)

# 'str' to 'int' = relaese_date
df['release_date'] = pd.to_numeric(df['release_date'], errors='coerce').fillna(0).astype(int)

print(df.columns.tolist())
print(df.release_date.head())

# Step 3.5: Statistical summary of the dataset

# this is done to check whether the features need scaling or not.
audio_features = ['danceability', 'energy', 'loudness', 'speechiness',
                  'acousticness', 'instrumentalness', 'liveness',
                  'valence', 'tempo', 'duration_ms']

print(df[audio_features].describe().round(3))

# from this we can see that the features need scaling.

# Step 3.6: Visualizing the distribution of audio features

# here we visually represent the distribution of each audio feature using histograms and boxplots to
# identify any skewness, outliers, or patterns in the data.

fig, axes = plt.subplots(3,4, figsize=(18,12))
axes = axes.flatten()

for i, col in enumerate(audio_features):
    axes[i].hist(df[col], bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5,
                    label=f"mean={df[col].mean():.2f}")
    axes[i].set_title(col, fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Count")
    axes[i].legend(fontsize=8)

for j in range(len(audio_features), len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("outputs/feature_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

# From the histograms, we can observe the distribution of each audio feature. 
# The red dashed lines indicate the mean values for each feature.

# Step 3.7: Correlation analysis

# here we compute the correlation matrix for the audio features and visualize it using a heatmap to identify 
# any strong correlations between features.


plt.figure(figsize=(14, 10))
correlation_matrix = df[audio_features].corr()

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    linewidths=0.5
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()

# This is a heatmap of the correlation matrix for the audio features.
# The colors indicate the strength and direction of the correlations, 
# with red representing positive correlations and blue representing negative correlations. 

# Step 3.8: Boxplots for outlier detection

# here we create boxplots for each audio feature to visually identify any outliers in the data.
# boxplots aregenerally useful for identifying outliers and understanding the spread of the data.

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
axes = axes.flatten()
for i, col in enumerate(audio_features):
    axes[i].boxplot(df[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[i].set_title(col, fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("outputs/boxplots_outliers.png", dpi=150, bbox_inches='tight')
plt.show()

# now we count the outliers using the IQR method.
# IQR = Q3-Q1
# lower_bound = Q1 - 1.5*IQR
# upper_bound = Q3 + 1.5*IQR

# Count outliers using IQR method
for col in audio_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower) | (df[col] > upper)].shape[0]
    percentage = round((outliers / len(df)) * 100, 1)
    print(f"{col} → {outliers} outliers ({percentage}%)")

# From the boxplots and the outlier counts, we can identify which features have a significant number of outliers.


# Step 4: Data saving

df.to_csv("cleaned_single_genre_artists.csv", index=False)
df_drop.to_csv("outputs/drop_columns.csv", index=False)

print("Saved cleaned_single_genre_artists.csv —", df.shape)
print("Saved drop_columns.csv —", df_drop.shape)

# Insights from the eda 

# our data has no missing values and no duplicates.
# features like speechiness and instrumentalness are extremely skewed.
# energy and loudness are strongly correlated at 0.73.
# valence and tempo are largely independent.




