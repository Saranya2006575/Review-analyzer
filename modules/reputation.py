"""
Reputation Score Engine
Calculates a multi-factor reputation score (0 to 100) combining:
- Normalized Average Rating (35%)
- Net Sentiment Distribution (30%)
- Positive to Negative Review Ratio (15%)
- Review Volume Confidence Factor (10%)
- Recent Review Trend Momentum (10%)
- Anomaly / Suspicious Activity Penalty (Deduction)
"""

import math
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


class ReputationEngine:
    """
    Computes an objective, multi-factor reputation index (0 - 100)
    with transparent mathematical auditing.
    """

    def __init__(
        self,
        w_rating: float = 0.35,
        w_sentiment: float = 0.30,
        w_ratio: float = 0.15,
        w_volume: float = 0.10,
        w_trend: float = 0.10
    ):
        self.w_rating = w_rating
        self.w_sentiment = w_sentiment
        self.w_ratio = w_ratio
        self.w_volume = w_volume
        self.w_trend = w_trend

    def calculate_reputation(
        self,
        df: pd.DataFrame,
        rating_col: str = 'Rating',
        sentiment_col: str = 'Sentiment',
        date_col: str = 'Date',
        suspicious_col: str = 'Is_Suspicious'
    ) -> Dict[str, Any]:
        """
        Executes reputation score formula across the provided DataFrame.
        """
        N = len(df)
        if N == 0:
            return {
                "reputation_score": 0.0,
                "grade": "N/A",
                "verdict": "No Data",
                "components": {},
                "metrics": {}
            }

        # 1. Average Rating Component (0 - 100)
        ratings = pd.to_numeric(df[rating_col], errors='coerce').dropna()
        avg_rating = float(ratings.mean()) if len(ratings) > 0 else 3.0
        s_rating = (avg_rating / 5.0) * 100.0

        # 2. Sentiment Distribution Component (0 - 100)
        sent_counts = df[sentiment_col].value_counts()
        pos_count = int(sent_counts.get('Positive', 0))
        neu_count = int(sent_counts.get('Neutral', 0))
        neg_count = int(sent_counts.get('Negative', 0))

        pct_pos = (pos_count / N) * 100.0
        pct_neu = (neu_count / N) * 100.0
        pct_neg = (neg_count / N) * 100.0

        # Net sentiment: range from -100 to +100 mapped to 0 - 100
        net_sentiment = pct_pos - pct_neg
        s_sentiment = (net_sentiment + 100.0) / 2.0

        # 3. Positive / Negative Ratio Component (0 - 100)
        if (pos_count + neg_count) > 0:
            s_ratio = (pos_count / (pos_count + neg_count)) * 100.0
        else:
            s_ratio = 50.0

        # 4. Volume Confidence Component (0 - 100)
        # Logarithmic confidence curve: 50+ reviews gives full 100% confidence
        conf_factor = min(1.0, math.log10(N + 1) / math.log10(50.0))
        s_volume = conf_factor * 100.0

        # 5. Recent Review Trend Momentum Component (0 - 100)
        s_trend = s_rating  # Default if date column is absent
        trend_direction = "Neutral"
        trend_delta = 0.0

        if date_col in df.columns:
            try:
                df_sorted = df.copy()
                df_sorted['dt'] = pd.to_datetime(df_sorted[date_col], errors='coerce')
                df_sorted = df_sorted.dropna(subset=['dt']).sort_values(by='dt')
                if len(df_sorted) >= 6:
                    split_idx = int(len(df_sorted) * 0.70)
                    hist_ratings = pd.to_numeric(df_sorted.iloc[:split_idx][rating_col], errors='coerce').dropna()
                    recent_ratings = pd.to_numeric(df_sorted.iloc[split_idx:][rating_col], errors='coerce').dropna()

                    hist_avg = float(hist_ratings.mean()) if len(hist_ratings) > 0 else avg_rating
                    recent_avg = float(recent_ratings.mean()) if len(recent_ratings) > 0 else avg_rating

                    trend_delta = round(recent_avg - hist_avg, 2)
                    if trend_delta > 0.15:
                        trend_direction = "Upward (+)"
                    elif trend_delta < -0.15:
                        trend_direction = "Downward (-)"
                    else:
                        trend_direction = "Stable"

                    # Map recent rating to 0-100 scale with momentum boost/drag
                    s_trend = min(100.0, max(0.0, (recent_avg / 5.0) * 100.0 + (trend_delta * 5.0)))
            except Exception:
                s_trend = s_rating

        # 6. Suspicious Activity Penalty
        penalty = 0.0
        flagged_count = 0
        if suspicious_col in df.columns:
            flagged_count = int(df[suspicious_col].sum())
            flagged_pct = (flagged_count / N) * 100.0
            # Deduct up to 15 points based on suspicious review volume
            penalty = min(15.0, flagged_pct * 0.35)

        # Weighted sum calculation
        raw_score = (
            (self.w_rating * s_rating) +
            (self.w_sentiment * s_sentiment) +
            (self.w_ratio * s_ratio) +
            (self.w_volume * s_volume) +
            (self.w_trend * s_trend)
        ) - penalty

        final_score = round(float(max(0.0, min(100.0, raw_score))), 1)

        # Grade & Verbal Verdict assignment
        if final_score >= 90.0:
            grade = "A+"
            verdict = "Stellar Reputation"
            color = "#10B981"  # Emerald
        elif final_score >= 80.0:
            grade = "A"
            verdict = "Strong Brand Reputation"
            color = "#3B82F6"  # Blue
        elif final_score >= 70.0:
            grade = "B"
            verdict = "Good Reputation"
            color = "#06B6D4"  # Cyan
        elif final_score >= 60.0:
            grade = "C"
            verdict = "Moderate / Fair Reputation"
            color = "#F59E0B"  # Amber
        elif final_score >= 45.0:
            grade = "D"
            verdict = "At-Risk Brand Reputation"
            color = "#F97316"  # Orange
        else:
            grade = "F"
            verdict = "Critical / Damaged Reputation"
            color = "#EF4444"  # Red

        return {
            "reputation_score": final_score,
            "grade": grade,
            "verdict": verdict,
            "color": color,
            "components": {
                "Rating Score (35%)": round(s_rating, 1),
                "Sentiment Score (30%)": round(s_sentiment, 1),
                "Pos/Neg Ratio Score (15%)": round(s_ratio, 1),
                "Volume Confidence (10%)": round(s_volume, 1),
                "Recent Trend Score (10%)": round(s_trend, 1),
                "Suspicious Activity Penalty": round(penalty, 1)
            },
            "metrics": {
                "total_reviews": N,
                "avg_rating": round(avg_rating, 2),
                "pct_positive": round(pct_pos, 1),
                "pct_neutral": round(pct_neu, 1),
                "pct_negative": round(pct_neg, 1),
                "pos_count": pos_count,
                "neu_count": neu_count,
                "neg_count": neg_count,
                "trend_direction": trend_direction,
                "trend_delta": trend_delta,
                "flagged_suspicious": flagged_count
            }
        }
