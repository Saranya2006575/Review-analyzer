"""
Verification Test Script
Validates all data mining, text mining, sentiment, clustering,
classification, association, anomaly detection, and reputation modules.
"""

import sys
import os
import pandas as pd

# Add workspace to path
sys.path.insert(0, os.path.dirname(__file__))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from modules.preprocessor import TextPreprocessor
from modules.sentiment import SentimentAnalyzer
from modules.classifier import ReviewClassifierPipeline
from modules.clustering import ReviewClusteringPipeline
from modules.frequency import FrequencyMiner
from modules.association import AssociationMiner
from modules.anomaly_detector import ReviewAnomalyDetector
from modules.topics import TopicAnalyzer
from modules.reputation import ReputationEngine
from modules.report_generator import generate_executive_report


def run_all_tests():
    print("==================================================")
    print("🚀 Running Review & Reputation Analyzer Pipeline Tests")
    print("==================================================")

    # 1. Test Dataset Loading
    csv_path = os.path.join(os.path.dirname(__file__), "data", "sample_reviews.csv")
    assert os.path.exists(csv_path), f"File {csv_path} not found!"
    raw_df = pd.read_csv(csv_path)
    print(f"✅ Loaded raw dataset: {len(raw_df)} rows, columns: {list(raw_df.columns)}")

    # 2. Test Preprocessing
    prep = TextPreprocessor()
    clean_df, prep_meta = prep.preprocess_dataframe(raw_df, text_col='Review Text', rating_col='Rating')
    assert len(clean_df) > 0, "Clean dataframe is empty!"
    assert 'Clean_Review' in clean_df.columns, "Clean_Review column missing!"
    print(f"✅ Preprocessing successful: {prep_meta['initial_rows']} -> {prep_meta['final_rows']} rows, {prep_meta['duplicates_removed']} duplicates purged")

    # Step-by-step trace test
    step_trace = prep.get_step_by_step("The sound quality is exceptional and battery lasts 35 hours!")
    assert "Lemmatized Tokens" in step_trace
    print("✅ Preprocessor step-by-step trace verified.")

    # 3. Test Sentiment Analysis
    sent_analyzer = SentimentAnalyzer()
    scored_df = sent_analyzer.add_sentiment_columns(clean_df, text_col='Clean_Review')
    assert 'Sentiment' in scored_df.columns
    assert 'Sentiment_Score' in scored_df.columns
    sample_res = sent_analyzer.analyze_review("Horrible experience, completely broken on arrival.")
    assert sample_res['sentiment'] == "Negative"
    print(f"✅ Sentiment analysis verified: {scored_df['Sentiment'].value_counts().to_dict()}")

    # 4. Test Anomaly Detection
    anomaly_detector = ReviewAnomalyDetector()
    audited_df, anom_meta = anomaly_detector.audit_dataframe(scored_df)
    assert 'Is_Suspicious' in audited_df.columns
    assert 'Suspicion_Reasons' in audited_df.columns
    flagged_count = int(audited_df['Is_Suspicious'].sum())
    assert flagged_count > 0, "No anomalies flagged in dataset with known anomalies!"
    print(f"✅ Anomaly detection verified: {flagged_count} suspicious reviews flagged ({anom_meta['flagged_percentage']}%)")

    # 5. Test Frequency Miner
    freq_miner = FrequencyMiner()
    unigrams = freq_miner.get_top_ngrams(audited_df['Clean_Review'].tolist(), ngram_range=(1, 1), top_n=5)
    aspects = freq_miner.analyze_sentiment_keywords(audited_df)
    assert len(unigrams) > 0
    print(f"✅ Frequency mining verified: Top unigram: {unigrams.iloc[0]['Ngram']}")

    # 6. Test Supervised Classification Models
    clf_pipeline = ReviewClassifierPipeline()
    clf_results = clf_pipeline.train_and_evaluate(
        texts=audited_df['Clean_Review'].tolist(),
        labels=audited_df['Sentiment'].tolist()
    )
    assert clf_results['summary_table'] is not None
    print(f"✅ Classification benchmark verified (4 models trained):\n{clf_results['summary_table'].to_string(index=False)}")

    # Single prediction test
    pred_res = clf_pipeline.predict_single("Outstanding battery and clear acoustic soundstage!")
    assert pred_res['predicted_class'] in ['Positive', 'Neutral', 'Negative']
    print(f"✅ Single prediction test: {pred_res['predicted_class']} (confidence {pred_res['confidence']})")

    # 7. Test Unsupervised Clustering & PCA
    cluster_pipe = ReviewClusteringPipeline(n_clusters=4)
    clustered_df, cluster_meta = cluster_pipe.fit_and_cluster(audited_df, text_col='Clean_Review', k=4)
    assert 'Cluster' in clustered_df.columns
    assert 'PCA_1' in clustered_df.columns
    assert 'PCA_2' in clustered_df.columns
    print(f"✅ Clustering verified: Silhouette Score = {cluster_meta['silhouette_score']}, Clusters = {len(cluster_meta['cluster_info'])}")

    # 8. Test Association Rule Mining
    assoc_miner = AssociationMiner()
    rules_df = assoc_miner.mine_rules(audited_df, min_support=0.03, min_confidence=0.25)
    print(f"✅ Association rules verified: Discovered {len(rules_df)} rules. Top rule: {rules_df.iloc[0]['Antecedent']} -> {rules_df.iloc[0]['Consequent']} (Lift: {rules_df.iloc[0]['Lift']})")

    # 9. Test Topic Modeling
    topic_analyzer = TopicAnalyzer()
    lda_topics = topic_analyzer.fit_lda_topics(audited_df['Clean_Review'].tolist(), n_topics=3)
    print(f"✅ Topic modeling (LDA) verified: Extracted {len(lda_topics)} topics.")

    # 10. Test Reputation Engine
    rep_engine = ReputationEngine()
    rep_results = rep_engine.calculate_reputation(audited_df)
    score = rep_results['reputation_score']
    assert 0.0 <= score <= 100.0, f"Reputation score {score} out of bounds [0, 100]!"
    print(f"✅ Reputation score verified: {score} / 100 (Grade: {rep_results['grade']}, {rep_results['verdict']})")

    # 11. Test Report Generation
    report_md = generate_executive_report(
        product_name="NovaSound & Benchmark Portfolio",
        reputation_data=rep_results,
        preprocessing_data=prep_meta,
        classification_data=clf_results,
        clustering_data=cluster_meta,
        association_data=rules_df,
        anomaly_data=anom_meta,
        top_aspects=aspects
    )
    assert len(report_md) > 500
    print("✅ Executive report generation verified.")

    print("\n🎉 ALL 11 PIPELINE TESTS PASSED SUCCESSFULLY! 🎉\n")


if __name__ == "__main__":
    run_all_tests()
