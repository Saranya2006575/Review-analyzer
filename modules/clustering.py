"""
Clustering Module
Implements unsupervised clustering techniques:
- TF-IDF text representation
- K-Means Clustering
- Automatic cluster theme labeling via top discriminating centroid terms
- PCA (Principal Component Analysis) 2D dimensionality reduction for interactive visual scatter
- Silhouette Score calculation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


class ReviewClusteringPipeline:
    """
    Unsupervised clustering pipeline for grouping customer reviews.
    """

    def __init__(self, n_clusters: int = 4, max_features: int = 1500):
        self.n_clusters = n_clusters
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            min_df=1
        )
        self.kmeans = None
        self.pca = PCA(n_components=2, random_state=42)
        self.cluster_themes = {}
        self.silhouette = 0.0

    def fit_and_cluster(
        self,
        df: pd.DataFrame,
        text_col: str = 'Clean_Review',
        k: int = 4
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Runs TF-IDF, K-Means clustering, PCA 2D reduction, and extracts top terms.
        """
        df_clustered = df.copy()
        raw_texts = df_clustered[text_col].fillna("").astype(str).tolist()

        # Adjust k if dataset is too small
        k = max(2, min(k, len(raw_texts) - 1))
        self.n_clusters = k

        # 1. TF-IDF
        X_tfidf = self.vectorizer.fit_transform(raw_texts)
        feature_names = np.array(self.vectorizer.get_feature_names_out())

        # 2. KMeans
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
        cluster_labels = self.kmeans.fit_predict(X_tfidf)
        df_clustered['Cluster'] = cluster_labels

        # 3. Silhouette score
        try:
            if len(df_clustered) > k and len(np.unique(cluster_labels)) > 1:
                self.silhouette = round(float(silhouette_score(X_tfidf, cluster_labels)), 3)
            else:
                self.silhouette = 0.0
        except Exception:
            self.silhouette = 0.0

        # 4. Extract Top Keywords per cluster
        centroids = self.kmeans.cluster_centers_
        cluster_info = {}

        for cluster_id in range(k):
            # Top 6 features with highest centroid weight
            top_indices = centroids[cluster_id].argsort()[::-1][:6]
            top_terms = [feature_names[i] for i in top_indices if i < len(feature_names)]
            theme_name = f"Cluster {cluster_id + 1}: {', '.join(top_terms[:3]).title()}"
            cluster_info[cluster_id] = {
                "theme_name": theme_name,
                "top_keywords": top_terms,
                "size": int((cluster_labels == cluster_id).sum()),
                "pct": round(float((cluster_labels == cluster_id).mean() * 100), 1)
            }

        self.cluster_themes = cluster_info
        df_clustered['Cluster_Theme'] = df_clustered['Cluster'].map(lambda c: cluster_info[c]['theme_name'])

        # 5. PCA 2D Projection for Interactive Plotly Visualization
        X_dense = X_tfidf.toarray()
        coords_2d = self.pca.fit_transform(X_dense)
        df_clustered['PCA_1'] = coords_2d[:, 0]
        df_clustered['PCA_2'] = coords_2d[:, 1]

        metadata = {
            "n_clusters": k,
            "silhouette_score": self.silhouette,
            "explained_variance_ratio": [round(float(v), 3) for v in self.pca.explained_variance_ratio_],
            "cluster_info": cluster_info
        }

        return df_clustered, metadata
