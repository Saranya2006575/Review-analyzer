"""
Classifier Module
Implements supervised Machine Learning classification algorithms:
- Multinomial Naive Bayes
- Logistic Regression
- Random Forest
- Support Vector Machine (LinearSVC)

Computes Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)


class ReviewClassifierPipeline:
    """
    Supervised Machine Learning pipeline for sentiment classification.
    Compares Naive Bayes, Logistic Regression, Random Forest, and SVM.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )
        self.models = {
            "Naive Bayes": MultinomialNB(alpha=0.5),
            "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42),
            "Support Vector Machine": LinearSVC(C=1.0, max_iter=2000, random_state=42)
        }
        self.trained_models = {}
        self.evaluation_results = {}
        self.classes = ['Negative', 'Neutral', 'Positive']
        self.is_trained = False

    def train_and_evaluate(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Fits vectorizer and trains all 4 classification algorithms.
        Computes Accuracy, Precision, Recall, F1, and Confusion Matrices.
        """
        clean_texts = [str(t) if t else "review" for t in texts]
        y = np.array(labels)

        # Ensure all 3 classes exist or balance split
        unique_classes = np.unique(y)
        stratify = y if len(unique_classes) > 1 and min(pd.Series(y).value_counts()) >= 2 else None

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            clean_texts, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

        X_train_vec = self.vectorizer.fit_transform(X_train_raw)
        X_test_vec = self.vectorizer.transform(X_test_raw)

        # Detect labels present in current dataset
        active_classes = [c for c in self.classes if c in unique_classes]
        if not active_classes:
            active_classes = list(unique_classes)

        results = {}
        for name, clf in self.models.items():
            # Clone / train model
            clf.fit(X_train_vec, y_train)
            self.trained_models[name] = clf

            # Predictions
            y_pred = clf.predict(X_test_vec)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            cm = confusion_matrix(y_test, y_pred, labels=active_classes)

            results[name] = {
                "accuracy": round(float(acc), 4),
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "confusion_matrix": cm.tolist(),
                "classes": active_classes,
                "report": classification_report(y_test, y_pred, zero_division=0, output_dict=True)
            }

        self.evaluation_results = results
        self.is_trained = True

        # Build comparison summary DataFrame
        summary_rows = []
        for name, m in results.items():
            summary_rows.append({
                "Algorithm": name,
                "Accuracy (%)": round(m["accuracy"] * 100, 2),
                "Precision (%)": round(m["precision"] * 100, 2),
                "Recall (%)": round(m["recall"] * 100, 2),
                "F1-Score (%)": round(m["f1_score"] * 100, 2),
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(by="F1-Score (%)", ascending=False).reset_index(drop=True)

        return {
            "summary_table": summary_df,
            "metrics": results,
            "train_size": len(X_train_raw),
            "test_size": len(X_test_raw),
            "feature_count": X_train_vec.shape[1]
        }

    def predict_single(self, text: str, model_name: str = "Logistic Regression") -> Dict[str, Any]:
        """
        Predicts sentiment category and confidence score for a new single review.
        """
        if not self.is_trained:
            return {"predicted_class": "Neutral", "confidence": 0.5, "probabilities": {}}

        clean_text = str(text) if text else "review"
        vec = self.vectorizer.transform([clean_text])
        clf = self.trained_models.get(model_name, self.trained_models.get("Logistic Regression"))

        predicted_class = clf.predict(vec)[0]

        probs = {}
        confidence = 1.0
        if hasattr(clf, "predict_proba"):
            p = clf.predict_proba(vec)[0]
            for cls_name, prob in zip(clf.classes_, p):
                probs[str(cls_name)] = round(float(prob), 3)
            confidence = round(float(max(p)), 3)
        elif hasattr(clf, "decision_function"):
            # LinearSVC decision function
            dec = clf.decision_function(vec)
            if dec.ndim > 1:
                # Softmax approximation
                exp_dec = np.exp(dec[0] - np.max(dec[0]))
                soft_p = exp_dec / np.sum(exp_dec)
                for cls_name, prob in zip(clf.classes_, soft_p):
                    probs[str(cls_name)] = round(float(prob), 3)
                confidence = round(float(max(soft_p)), 3)
            else:
                prob_pos = 1 / (1 + np.exp(-dec[0]))
                probs = {"Negative": round(1 - prob_pos, 3), "Positive": round(prob_pos, 3)}
                confidence = round(float(max(prob_pos, 1 - prob_pos)), 3)

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probs,
            "model_used": model_name
        }
