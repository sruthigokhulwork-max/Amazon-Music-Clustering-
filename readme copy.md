
AMAZON MUSIC CLUSTERING

1. PROJECT OVERVIEW
 The project is about grouping 95,837 songs based on its audio characteristics using unsupervised machine learning.

PROJECT STRUCTURE
1. eda_analysis.py
2. preprocessing.py
3. clustering.py
4. visualization.py
5. readme.md
6. outputs
   - feature_distributions.png
   - correlation_heatmap.png
   - boxplots_outliers.png
   - scaling_comparison.png
   - pca_variance.png
   - pca_2.png
   - elbow_plot.png
   - silhouette_plot.png
   - pca_clusters_colored.png
   - cluster_feature_bars.png
   - cluster_heatmap.png
   - final_clustered_songs.csv
   - clustered_songs.csv
   - clustered_features.csv
   - x_scaled.csv
   - x_pca_2d.csv
   - drop_column.csv

2. Our approach

"EDA ANALYSIS"
- firstly, we loaded the dataset for our eda analysins
- while exploring we found no null, duplicates or any missing values in our dataset.
- since clustering was the main focus , we had no use in using the str values so we copied them first for later use and then dropped them.
- we converted the string values of some to numericals as we need them for clustering.
- we did statstical analysis of our dataset to check whether it needed scaling or not.
- from the audio features we found out our data needed scaling.
- we did the visualization using histograms and boxplots to identify any skewness, outliers, or patterns in the data and we found that some were skewed.
- correlation analyis was done using the heatmap to identify any strong correlations between features.
- outliers were detected using box plots and they were counted using the IQR method.
<!-- insights -->
# our data had no missing values and no duplicates.
# features like speechiness and instrumentalness were extremely skewed.
# energy and loudness were strongly correlated at 0.73.
# valence and tempo were largely independent.

"PREPROCESSING"
- preprocessing was done to prepare our dataset for clustering using feature selection, standardscalar and PCA.
- after selecting the audio features, we selected only the columns that were used for clustering(clustering_features).
- after feature selection, we done the scaling using the standard scalar. we visualzied them using histograms.
- we particulary used only 5 columns to verify skewness those 5 represented the most extreme range differences in the dataset.
- pca analysis was done to create a 2 principal components and then they were converted into a 2d.
<!-- insights -->
# before scaling loudness went from -50 to 0. After scaling it went from -8 to +2. Before scaling duration_ms went from 0 to 5 
# million. After scaling it went from 0 to 40. all the features were in comparable range.
# PC1 alone = 21.3% of information. PC1 + PC2 together = 35.9%.
# from 2d pca,the scattered dots going all the way down to -15 on the y-axis and -10 on the x-axis are our outlier songs.

"CLUSTERING"
-  it meant that an algorithm that groups data points into K clusters by minimizing the distance between points and their group's center.
- clustering: elbow method,silhoutte score
- using the elbow method we found that our graph was a gradual curve with no sharp elbow but from k=2 to k=5 we could see our curve dropped steeply and then from k=5 till k=15 dropped gradually so the bend happened from k=5.
- from the silhoutte score,# from this we understood that highest score was at K=3, dropped sharply at K=4 and K=5, 
- then became flat and low from K=6 onwards.even though k=3 had the highest score, having only 3 clusters for our dataset made it too broad, it was not useful.
-k=4 is the 2nd best and was nearer to the elbow and having extra cluster was useful
- from that, we found that k = 4 is the best K.
- we applied the best k and evaluated the results.
- we did the dbscan 
<!-- insights -->
# comparing our elbow method and silhoutte score with the pca plot we could see the main dense cloud split into 2-3 colored # # regions, and the outlier dots at the bottom left forming their own separate colored group. 
# That separation shows our clustering is fine.
# DBSCAN found    : 63 clusters - 3 clusters is way too many for 95,837 songs and is not useful for music categorization.
# Noise points    : 6791 (7.1%) - these are outliers but labeling 6,791 songs as uncategorizable is not useful for a music recommendation system.
# K-Means K=4 Silhouette : 0.1669
# Conclusion      : K-Means chosen as final model

"VISUALIZATION"
- we defined the audio features cluster names.
- we did cluster profiling.
- to know the how our features were different, we calculated using clusters(bar chart)
- to normalize the profiles, we calculated using clusters(heat map)
- we validated the genre.
- we found top 5 tracks per cluster.
<!-- insights -->
# Evaluation Metrics:
# Silhouette Score : 0.1669
# Davies-Bouldin Index : 1.8906
# Inertia : 874,772

# cluster results:
# Cluster 0 — Calm Acoustic Songs (29,239 songs — 30.5%)
# High acousticness (0.70), low energy (0.34), quiet (-12.1 dB).
# Folk, classical, and soft acoustic tracks.

# Cluster 1 — High Energy Mainstream (46,448 songs — 48.5%)
# Highest energy (0.71), loudest (-7.43 dB), happiest (valence 0.68).
# Pop, rock, and electronic mainstream music.

# Cluster 2 — Speech Heavy & Live (12,335 songs — 12.9%)
# Speechiness 0.84 — 14x higher than other clusters.
# Shortest songs (97,081ms avg). Rap, spoken word, live recordings.

# Cluster 3 — Pure Instrumental (7,815 songs — 8.2%)
# Instrumentalness 0.81 — almost no vocals whatsoever.
# Classical, ambient, film scores, jazz instrumentals.


 <!-- Key Insight -->

- Speechiness and instrumentalness were the strongest separators.
- K=4 was confirmed by both the Elbow Method and Silhouette Score. 
- DBSCAN comparison (63 clusters, 7.1% noise) further validated 
that K-Means was the right algorithm for this dataset.
- The resulting clusters can directly power real-world applications, personalized playlist generation, song recommendations artist market analysis, and user segmentation.
