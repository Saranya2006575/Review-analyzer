"""
Report Generator Module
Compiles data mining discoveries, statistical distributions, anomaly audits,
and reputation metrics into a clean, comprehensive executive Markdown report.
"""

from datetime import datetime
from typing import Dict, Any, List
import pandas as pd


def generate_executive_report(
    product_name: str,
    reputation_data: Dict[str, Any],
    preprocessing_data: Dict[str, Any],
    classification_data: Dict[str, Any],
    clustering_data: Dict[str, Any],
    association_data: pd.DataFrame,
    anomaly_data: Dict[str, Any],
    top_aspects: Dict[str, Any]
) -> str:
    """
    Constructs a structured executive reputation report.
    """
    now_str = datetime.now().strftime("%B %d, %Y - %H:%M")
    rep_score = reputation_data.get("reputation_score", 0.0)
    grade = reputation_data.get("grade", "N/A")
    verdict = reputation_data.get("verdict", "N/A")
    metrics = reputation_data.get("metrics", {})
    components = reputation_data.get("components", {})

    total_reviews = metrics.get("total_reviews", 0)
    avg_rating = metrics.get("avg_rating", 0.0)
    pct_pos = metrics.get("pct_positive", 0.0)
    pct_neu = metrics.get("pct_neutral", 0.0)
    pct_neg = metrics.get("pct_negative", 0.0)
    trend_dir = metrics.get("trend_direction", "Stable")
    trend_delta = metrics.get("trend_delta", 0.0)
    flagged_suspicious = metrics.get("flagged_suspicious", 0)

    # Best classification model
    summary_df = classification_data.get("summary_table", pd.DataFrame())
    best_model_name = summary_df.iloc[0]["Algorithm"] if len(summary_df) > 0 else "Logistic Regression"
    best_f1 = summary_df.iloc[0]["F1-Score (%)"] if len(summary_df) > 0 else 0.0
    best_acc = summary_df.iloc[0]["Accuracy (%)"] if len(summary_df) > 0 else 0.0

    # Clusters
    clusters = clustering_data.get("cluster_info", {})

    # Top Association Rules
    top_rules_text = ""
    if isinstance(association_data, pd.DataFrame) and len(association_data) > 0:
        top_rules_text = "| Antecedent Aspect | Consequent Outcome | Support | Confidence | Lift |\n|---|---|---|---|---|\n"
        for _, r in association_data.head(5).iterrows():
            top_rules_text += f"| **{r.get('Antecedent')}** | {r.get('Consequent')} | {r.get('Support (%)')}% | {r.get('Confidence (%)')}% | **{r.get('Lift')}x** |\n"
    else:
        top_rules_text = "_No strong association patterns reached minimum support threshold._\n"

    # Praised vs Complaints
    praised = top_aspects.get("praised_aspects", pd.DataFrame())
    complaints = top_aspects.get("common_complaints", pd.DataFrame())

    praised_list = ", ".join([f"`{r['Ngram']}`" for _, r in praised.head(6).iterrows()]) if len(praised) > 0 else "None identified"
    complaints_list = ", ".join([f"`{r['Ngram']}`" for _, r in complaints.head(6).iterrows()]) if len(complaints) > 0 else "None identified"

    report = f"""# Executive Reputation & Data Mining Audit Report

**Target Subject**: {product_name}  
**Audit Date**: {now_str}  
**Auditing Framework**: Data Mining (KDD) & Computational Text Mining Pipeline  
**Dataset Size**: {total_reviews} Reviews Audited  

---

## 1. Executive Summary & Reputation Scorecard

| Overall Reputation Index | Letter Grade | Verbal Assessment | Recent Momentum Trend |
| :---: | :---: | :---: | :---: |
| # **{rep_score} / 100** | # **{grade}** | **{verdict}** | **{trend_dir} ({'+' if trend_delta > 0 else ''}{trend_delta}★)** |

### Reputation Breakdown Factors:
- **Normalized Rating Contribution (35% weight)**: {components.get('Rating Score (35%)', 0)} / 100
- **Net Sentiment Polarity (30% weight)**: {components.get('Sentiment Score (30%)', 0)} / 100
- **Positive-to-Negative Review Ratio (15% weight)**: {components.get('Pos/Neg Ratio Score (15%)', 0)} / 100
- **Volume Confidence Adjustment (10% weight)**: {components.get('Volume Confidence (10%)', 0)} / 100
- **Time-Decay Recent Trend Momentum (10% weight)**: {components.get('Recent Trend Score (10%)', 0)} / 100
- **Suspicious Review Penalty**: -{components.get('Suspicious Activity Penalty', 0)} pts

---

## 2. Customer Sentiment & Star Rating Distribution

- **Average Star Rating**: {avg_rating} / 5.00 Stars
- **Positive Feedback**: {pct_pos}% ({metrics.get('pos_count', 0)} reviews)
- **Neutral Feedback**: {pct_neu}% ({metrics.get('neu_count', 0)} reviews)
- **Negative Feedback**: {pct_neg}% ({metrics.get('neg_count', 0)} reviews)

---

## 3. Data Preprocessing & Hygiene Statistics

- **Raw Reviews Input**: {preprocessing_data.get('initial_rows', 0)}
- **Null Reviews Imputed/Handled**: {preprocessing_data.get('nulls_in_text', 0)}
- **Identical Duplicates Purged**: {preprocessing_data.get('duplicates_removed', 0)}
- **Vocabulary Size Reduction after Stopwords & Lemmatization**: {preprocessing_data.get('vocab_reduction_pct', 0)}%

---

## 4. Supervised Machine Learning Benchmarks

Four supervised text classification algorithms were trained on the TF-IDF vectorized review corpus using stratified 80/20 train/test partitions:

- **Champion Model**: {best_model_name}
- **Accuracy**: {best_acc}%
- **Weighted F1-Score**: {best_f1}%

---

## 5. Unsupervised Clustering & Thematic Segmentation

K-Means clustering partitioned customer sentiment into thematic clusters (Silhouette Score: {clustering_data.get('silhouette_score', 'N/A')}):

"""
    for cid, cdata in clusters.items():
        report += f"- **{cdata.get('theme_name')}** ({cdata.get('size')} reviews, {cdata.get('pct')}% of corpus)\n"
        report += f"  - _Salient Terms_: {', '.join(cdata.get('top_keywords', []))}\n"

    report += f"""
---

## 6. Association Rule Mining (Aspect-to-Sentiment Correlations)

Using frequent itemset mining, the following high-lift association rules were discovered linking customer experience dimensions to outcomes:

{top_rules_text}

---

## 7. Anomaly & Suspicious Activity Audit

- **Reviews Flagged as Potentially Suspicious**: {flagged_suspicious} ({anomaly_data.get('flagged_percentage', 0)}% of dataset)
- **Duplicate / Highly Similar Spam Clusters**: {anomaly_data.get('anomaly_breakdown', {}).get('duplicate_similarity', 0)}
- **Excessive Repetition / Keyword Stuffing**: {anomaly_data.get('anomaly_breakdown', {}).get('lexical_repetition', 0)}
- **Rating vs Text Sentiment Contradictions**: {anomaly_data.get('anomaly_breakdown', {}).get('rating_sentiment_mismatch', 0)}
- **Length Anomalies (Unusually Short/Long)**: {anomaly_data.get('anomaly_breakdown', {}).get('length_anomalies', 0)}

> _Note: Reviews are classified under probabilistic anomaly criteria and marked as "Potentially Suspicious" for human-in-the-loop review rather than asserting fraud._

---

## 8. Aspect Insights & Strategic Action Items

- **Top Praised Features / Keywords**:
  {praised_list}
- **Primary Customer Pain Points**:
  {complaints_list}

### Recommended Strategic Next Steps:
1. **Target Dominant Negative Association**: Prioritize engineering or customer service fixes for aspects showing strong correlation with 1-star ratings.
2. **Review Integrity Monitoring**: Review flagged anomalous accounts to prevent astroturfing or automated bot reviews.
3. **Sentiment Velocity Tracking**: Capitalize on praised aspects in marketing campaigns while resolving negative cluster concerns.

---
_Generated automatically by the Review and Reputation Analyzer Web Application._
"""
    return report
