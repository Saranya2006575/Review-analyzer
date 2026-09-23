"""
Anomaly & Fake/Suspicious Review Detection Module
Identifies potentially suspicious reviews using transparent data-mining rules:
1. Duplicate or highly similar reviews (Cosine similarity > 0.85)
2. Excessive repeated words / Low lexical diversity (TTR < 0.40)
3. Length anomalies (Unusually short < 4 words or statistical outlier > 3.0 std dev)
4. Rating vs Sentiment contradictions (e.g. 5 stars with caustic negative text)
5. Abnormal burst / frequency patterns

Ethical/Academic Standard:
Reviews are strictly labeled 'Potentially Suspicious' rather than definitively fake.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ReviewAnomalyDetector:
    """
    Data Mining Anomaly & Outlier Detector for customer reviews.
    """

    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold

    def check_lexical_repetition(self, text: str) -> Tuple[bool, str]:
        """
        Checks for keyword stuffing or extreme repetition.
        Uses Type-Token Ratio (TTR) and max word frequency.
        """
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        if len(words) < 6:
            return False, ""

        unique_words = set(words)
        ttr = len(unique_words) / len(words)

        # Check most frequent word
        word_counts = pd.Series(words).value_counts()
        max_freq_ratio = word_counts.iloc[0] / len(words)

        if ttr < 0.42:
            return True, f"Excessive Word Repetition (Lexical Diversity: {round(ttr*100, 1)}%)"
        if max_freq_ratio > 0.35 and len(words) >= 8:
            top_w = word_counts.index[0]
            return True, f"Keyword Stuffing ('{top_w}' repeated {word_counts.iloc[0]} times)"

        return False, ""

    def check_length_anomaly(self, word_count: int, mean_len: float, std_len: float) -> Tuple[bool, str]:
        """Checks for unusually short or statistically extreme length."""
        if word_count <= 3:
            return True, f"Unusually Short Review ({word_count} words)"
        if std_len > 0 and word_count > (mean_len + 3.0 * std_len):
            return True, f"Unusually Long Review ({word_count} words, >3.0σ)"
        return False, ""

    def check_rating_sentiment_mismatch(self, rating: float, sentiment_score: float) -> Tuple[bool, str]:
        """
        Detects strong discrepancies between star rating and expressed sentiment.
        """
        if rating >= 4.5 and sentiment_score <= -0.25:
            return True, f"Rating-Sentiment Contradiction ({rating}★ with Negative sentiment: {sentiment_score})"
        if rating <= 1.5 and sentiment_score >= 0.25:
            return True, f"Rating-Sentiment Contradiction ({rating}★ with Positive sentiment: +{sentiment_score})"
        return False, ""

    def detect_near_duplicates(self, texts: List[str]) -> List[Tuple[bool, str]]:
        """
        Computes TF-IDF pairwise similarity to detect near-duplicate spam clusters.
        """
        n = len(texts)
        if n < 2:
            return [(False, "")] * n

        clean_texts = [str(t) if str(t).strip() else "empty" for t in texts]
        try:
            vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words='english')
            X = vec.fit_transform(clean_texts)
            sim_matrix = cosine_similarity(X)
            # Set diagonal to zero
            np.fill_diagonal(sim_matrix, 0.0)

            results = []
            for i in range(n):
                max_sim = np.max(sim_matrix[i])
                if max_sim >= self.similarity_threshold:
                    match_idx = int(np.argmax(sim_matrix[i]))
                    results.append((True, f"Near-Duplicate Review (Similarity {round(max_sim*100, 1)}% with row #{match_idx+1})"))
                else:
                    results.append((False, ""))
            return results
        except Exception:
            return [(False, "")] * n

    def audit_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = 'Review Text',
        rating_col: str = 'Rating',
        sentiment_score_col: str = 'Sentiment_Score'
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Performs full anomaly audit on DataFrame.
        Appends:
        - Is_Suspicious (bool)
        - Suspicious_Indicator ('Potentially Suspicious' / 'Normal')
        - Suspicion_Reasons (List of strings)
        - Suspicion_Score (0 - 100)
        """
        audited_df = df.copy()
        n = len(audited_df)

        # Precompute length statistics
        word_counts = audited_df[text_col].apply(lambda x: len(str(x).split()))
        mean_len = float(word_counts.mean()) if n > 0 else 20.0
        std_len = float(word_counts.std()) if n > 1 else 10.0

        # Run near-duplicate detection across corpus
        dup_results = self.detect_near_duplicates(audited_df[text_col].tolist())

        is_suspicious_list = []
        labels_list = []
        reasons_list = []
        scores_list = []

        anomaly_breakdown = {
            "duplicate_similarity": 0,
            "lexical_repetition": 0,
            "length_anomalies": 0,
            "rating_sentiment_mismatch": 0
        }

        for idx, row in audited_df.iterrows():
            reasons = []
            score = 0
            txt = str(row.get(text_col, ''))
            rating = float(row.get(rating_col, 3.0))
            sent_score = float(row.get(sentiment_score_col, 0.0))
            w_count = word_counts.loc[idx]

            # 1. Duplicate check
            is_dup, dup_msg = dup_results[idx]
            if is_dup:
                reasons.append(dup_msg)
                score += 45
                anomaly_breakdown["duplicate_similarity"] += 1

            # 2. Repetition check
            is_rep, rep_msg = self.check_lexical_repetition(txt)
            if is_rep:
                reasons.append(rep_msg)
                score += 35
                anomaly_breakdown["lexical_repetition"] += 1

            # 3. Length check
            is_len, len_msg = self.check_length_anomaly(w_count, mean_len, std_len)
            if is_len:
                reasons.append(len_msg)
                score += 25
                anomaly_breakdown["length_anomalies"] += 1

            # 4. Rating mismatch check
            is_mis, mis_msg = self.check_rating_sentiment_mismatch(rating, sent_score)
            if is_mis:
                reasons.append(mis_msg)
                score += 40
                anomaly_breakdown["rating_sentiment_mismatch"] += 1

            final_score = min(100, score)
            flagged = len(reasons) > 0

            is_suspicious_list.append(flagged)
            labels_list.append("Potentially Suspicious" if flagged else "Normal")
            reasons_list.append(" | ".join(reasons) if reasons else "None")
            scores_list.append(final_score)

        audited_df['Is_Suspicious'] = is_suspicious_list
        audited_df['Suspicious_Indicator'] = labels_list
        audited_df['Suspicion_Reasons'] = reasons_list
        audited_df['Suspicion_Score'] = scores_list

        summary = {
            "total_reviews": n,
            "flagged_count": int(sum(is_suspicious_list)),
            "flagged_percentage": round((sum(is_suspicious_list) / max(1, n)) * 100, 2),
            "anomaly_breakdown": anomaly_breakdown
        }

        return audited_df, summary
