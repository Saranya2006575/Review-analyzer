"""
Topic Modeling and Aspect Analysis Module
Extracts latent discussion topics using Latent Dirichlet Allocation (LDA)
and classifies praised features vs common customer complaints.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


class TopicAnalyzer:
    """
    Topic Modeling via LDA and Praised vs Complaint Aspect extraction.
    """

    def __init__(self, n_topics: int = 4):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_df=0.90,
            min_df=1,
            stop_words='english',
            max_features=1200
        )
        self.lda_model = None

    def fit_lda_topics(self, texts: List[str], n_topics: int = 4, words_per_topic: int = 6) -> List[Dict[str, Any]]:
        """
        Discovers latent topics using unsupervised LDA.
        """
        valid_texts = [str(t) for t in texts if str(t).strip()]
        if len(valid_texts) < n_topics:
            n_topics = max(1, len(valid_texts))

        self.n_topics = n_topics
        self.lda_model = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            learning_method='online',
            max_iter=25
        )

        try:
            doc_term_matrix = self.vectorizer.fit_transform(valid_texts)
            self.lda_model.fit(doc_term_matrix)
            feature_names = self.vectorizer.get_feature_names_out()

            topics = []
            for topic_idx, topic in enumerate(self.lda_model.components_):
                top_indices = topic.argsort()[:-words_per_topic - 1:-1]
                top_words = [feature_names[i] for i in top_indices]
                topic_title = f"Topic {topic_idx + 1}: {', '.join(top_words[:3]).title()}"
                topics.append({
                    "topic_id": topic_idx + 1,
                    "title": topic_title,
                    "top_words": top_words,
                    "word_weights": [round(float(topic[i]), 2) for i in top_indices]
                })
            return topics
        except Exception:
            return []

    def extract_review_aspect_tags(self, text: str) -> str:
        """
        Extracts salient aspect tags present in a single review.
        """
        keywords_pool = {
            "Audio": ["sound", "audio", "bass", "treble", "noise", "anc", "mic"],
            "Battery": ["battery", "charge", "charging", "drain", "power"],
            "Comfort": ["comfort", "cushion", "lightweight", "earcup", "strap"],
            "Build": ["build", "durability", "titanium", "plastic", "hinge"],
            "Service": ["support", "service", "warranty", "refund", "agent"],
            "App/Sync": ["app", "bluetooth", "sync", "connection", "firmware"],
            "Delivery": ["delivery", "driver", "courier", "package", "packaging"],
            "Price": ["price", "cost", "subscription", "worth", "tier"]
        }
        low = text.lower()
        matched = []
        for cat, kw_list in keywords_pool.items():
            if any(k in low for k in kw_list):
                matched.append(cat)

        return ", ".join(matched) if matched else "General"
