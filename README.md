# 🛡️ Customer Review & Reputation Analyzer
### A Complete Data Mining & Text Mining Web Application

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Visuals-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)

---

## 📌 Project Overview

The **Review & Reputation Analyzer** is an end-to-end web application engineered to demonstrate the full spectrum of **Data Mining (Knowledge Discovery in Databases - KDD)** and **Computational Text Mining** methodologies.

Unlike simple sentiment analysis tools that only output positive or negative labels, this application performs comprehensive **pattern discovery, cluster profiling, association rule mining, anomaly/spam detection, supervised machine learning benchmarking, and multi-factor reputation index calculation (0–100)**.

---

## 🏗️ Project Folder Structure

```
pbl/
├── app.py                      # Main Streamlit web application & page router
├── requirements.txt            # Python dependencies (Streamlit, Pandas, Sklearn, etc.)
├── test_pipeline.py            # Automated test suite verifying all 11 backend modules
├── sample_report.md            # Sample final executive reputation audit report
├── README.md                   # Comprehensive documentation & theory guide
├── data/
│   └── sample_reviews.csv      # Rich benchmark dataset (200+ realistic reviews across 4 brands)
└── modules/
    ├── __init__.py             # Module initialization
    ├── preprocessor.py         # Text cleaning, regex, tokenization, lemmatization, deduplication
    ├── sentiment.py            # Hybrid VADER + rule-based lexicon sentiment scoring engine
    ├── classifier.py           # Supervised classification models (NB, LogReg, RF, SVM) & metrics
    ├── clustering.py           # K-Means clustering, PCA 2D projection, cluster theme extraction
    ├── frequency.py            # N-Gram frequency mining, positive/negative keyword distributions
    ├── association.py          # Association rule & frequent itemset mining (Support, Confidence, Lift)
    ├── anomaly_detector.py     # Suspicious review detection (near-duplicates, repetition, mismatch)
    ├── topics.py               # Latent Dirichlet Allocation (LDA) topic modeling & aspect extraction
    ├── reputation.py           # Multi-factor 0-100 reputation score engine with penalty damping
    └── report_generator.py     # Executive Markdown report compiler
```

---

## 🚀 Installation & Setup Instructions

### 1. Prerequisites
- Python 3.10, 3.11, or newer installed on your machine.
- Terminal / PowerShell / Command Prompt.

### 2. Clone / Navigate to Directory
```bash
cd path/to/pbl
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: If NLTK data is downloaded for the first time, the preprocessor will automatically download `punkt`, `stopwords`, and `wordnet` in the background with zero configuration required).*

### 4. Run Automated Tests
Verify that all data mining modules and algorithms execute properly:
```bash
python test_pipeline.py
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
After running this command, open your browser at `http://localhost:8501`.

---

## 🔬 Data Mining & Text Mining Techniques Used

### 1. Text Preprocessing & Cleaning (`modules/preprocessor.py`)
- **Missing Value Handling**: Imputes missing ratings with the column median; drops records lacking textual content.
- **Deduplication**: Detects identical textual submissions from repetitive submissions.
- **Normalization**: Strips HTML tags, email addresses, URLs, digits, and special symbols; case-folding to lowercase.
- **Tokenization**: Segmenting sentences into discrete lexical tokens.
- **Stop Words Filtering**: Purging non-informative English stop words (`the`, `and`, `is`, `for`, etc.).
- **Lemmatization**: Morphological reduction of tokens to their base dictionary lemma using WordNet (e.g. *cancelling* $\to$ *cancel*, *earcups* $\to$ *earcup*).

### 2. Supervised Sentiment Classification (`modules/classifier.py`)
Vectorizes cleaned text using **Term Frequency - Inverse Document Frequency (TF-IDF)** with sublinear term-frequency scaling and unigram/bigram token pairs:
- **Multinomial Naive Bayes**: Fast probabilistic classifier based on Bayes' theorem with Laplace smoothing.
- **Logistic Regression**: Linear discriminative model using cross-entropy loss with L2 regularization.
- **Random Forest**: Ensemble bagging classifier aggregating 100 decision trees to curb variance.
- **Support Vector Machine (LinearSVC)**: Finds optimal maximum-margin separating hyper-planes in high-dimensional text space.
- **Evaluation**: Computes stratified 80/20 train/test metrics:
  - $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
  - $\text{Precision} = \frac{TP}{TP + FP}$
  - $\text{Recall} = \frac{TP}{TP + FN}$
  - $\text{F1-Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - **Interactive Confusion Matrix Heatmap** displaying true vs predicted class distributions.

### 3. Unsupervised Clustering & Projection (`modules/clustering.py`)
- **K-Means Clustering**: Partitions reviews into $k$ thematic clusters by iteratively minimizing Within-Cluster Sum of Squares (WCSS):
  $$\min \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$
- **Cluster Theme Extraction**: Identifies highest centroid TF-IDF coordinates to dynamically name each cluster (e.g., *Cluster 1: Noise Cancellation Fidelity*).
- **Principal Component Analysis (PCA)**: Reduces the sparse 2,000-dimensional TF-IDF space into 2 orthogonal principal components for interactive 2D scatter visualization.
- **Silhouette Coefficient**: Quantifies cluster separation and cohesion:
  $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

### 4. Frequency Mining & N-Gram Extraction (`modules/frequency.py`)
- Extracts most prominent **Unigrams**, **Bigrams**, and **Trigrams** separated into positive and negative customer subsets.
- Uncovers distinct linguistic praise vs complaint collocations (e.g., *"battery life"*, *"sound quality"* vs *"headband plastic"*, *"customer support"*).

### 5. Association Rule Mining (`modules/association.py`)
Translates each review into a transaction basket of functional aspects, sentiment polarities, and rating tiers. Uncovers relational association rules ($A \implies B$):
- **Support**: Probability that both antecedent $A$ and consequent $B$ appear together:
  $$\text{Support}(A \implies B) = P(A \cap B) = \frac{\text{count}(A \cup B)}{N}$$
- **Confidence**: Conditional probability of outcome $B$ given antecedent aspect $A$:
  $$\text{Confidence}(A \implies B) = P(B \mid A) = \frac{\text{count}(A \cup B)}{\text{count}(A)}$$
- **Lift**: Measure of rule strength relative to random chance ($> 1.0$ indicates positive correlation):
  $$\text{Lift}(A \implies B) = \frac{\text{Confidence}(A \implies B)}{\text{Support}(B)}$$

### 6. Anomaly & Suspicious Review Detection (`modules/anomaly_detector.py`)
Flags potentially suspicious reviews using multi-criteria probabilistic rules:
1. **Cosine Similarity Near-Duplicates**: Pairwise TF-IDF cosine similarity $\ge 0.85$ detects automated copy-paste astroturfing.
2. **Lexical Repetition (Type-Token Ratio)**: $TTR = \frac{|\text{unique tokens}|}{|\text{total tokens}|} < 0.42$ catches keyword stuffing or spam repetition.
3. **Length Anomalies**: Reviews with $\le 3$ words or length exceeding $\mu + 3.0\sigma$.
4. **Rating vs Sentiment Mismatch**: Discrepancies such as a 5-star rating with strongly negative text ($S \le -0.25$) or a 1-star rating with glowing positive text ($S \ge +0.25$).
- **Academic Responsibility**: Reviews are labeled **"Potentially Suspicious"** rather than claiming definitive fraud, enabling human-in-the-loop review.

---

## 🧮 Mathematical Formulation of Reputation Score ($R$)

The **Reputation Score** ranges from **0 to 100**, synthesizing five core operational factors and an anomaly penalty:

$$R = \max\left(0, \min\left(100, \sum_{i=1}^{5} w_i \cdot S_i - P_{\text{suspicious}}\right)\right)$$

### 1. Factor Weights:
| Factor | Symbol | Weight ($w_i$) | Purpose |
|---|:---:|:---:|---|
| Normalized Star Rating | $S_1$ | **35%** | Base baseline satisfaction metric |
| Net Sentiment Distribution | $S_2$ | **30%** | Polarity balance of text emotions |
| Positive-to-Negative Ratio | $S_3$ | **15%** | Proportion of advocates vs detractors |
| Review Volume Confidence | $S_4$ | **10%** | Statistical reliability dampening |
| Recent Trend Momentum | $S_5$ | **10%** | Temporal velocity / improvement trajectory |

### 2. Formulas:
- **Normalized Rating ($S_1$)**:
  $$S_1 = \left(\frac{\mu_{\text{rating}}}{5.0}\right) \times 100$$
- **Net Sentiment Polarity ($S_2$)**:
  $$S_2 = \frac{\%Pos - \%Neg + 100}{2} \in [0, 100]$$
- **Positive-to-Negative Ratio ($S_3$)**:
  $$S_3 = \frac{\text{Count}(Pos)}{\text{Count}(Pos) + \text{Count}(Neg)} \times 100$$
- **Volume Reliability Confidence ($S_4$)**:
  Logarithmic saturation curve ensuring small samples don't produce artificially inflated scores:
  $$S_4 = \min\left(1.0, \frac{\log_{10}(N + 1)}{\log_{10}(50)}\right) \times 100$$
- **Recent Trend Momentum ($S_5$)**:
  Compares rating average of the recent 30% of reviews against historical baseline to reward upward momentum.
- **Suspicious Review Penalty ($P_{\text{suspicious}}$)**:
  $$P_{\text{suspicious}} = \min(15.0, \text{Flagged\_Pct} \times 0.35)$$

### 3. Letter Grade & Verdict System:
- **90 – 100**: **A+** (Stellar Reputation)
- **80 – 89**: **A** (Strong Brand Reputation)
- **70 – 79**: **B** (Good Reputation)
- **60 – 69**: **C** (Moderate / Fair Reputation)
- **45 – 59**: **D** (At-Risk Brand Reputation)
- **< 45**: **F** (Critical / Damaged Reputation)

---

## 📊 Dataset Attributes Supported (`sample_reviews.csv`)

| Column Name | Type | Description |
|---|---|---|
| `Review ID` | String | Unique review identifier (e.g. `REV-1001`) |
| `Customer Name` | String | Full name or user handle of reviewer |
| `Rating` | Integer (1–5) | Numerical star rating |
| `Review Text` | String | Verbatim customer feedback text |
| `Date` | Date (YYYY-MM-DD) | Submission timestamp |
| `Product/Business Name` | String | Product or service name |

The benchmark dataset includes 200+ realistic entries across consumer electronics, cloud software, food delivery, and fitness wearables, as well as intentional edge-case anomalies for testing.

---

## 💻 Streamlit UI Walkthrough

1. **📊 Executive Dashboard**: High-level scorecards, gauge chart, star distribution bar charts, sentiment donut chart, time-series momentum graph, and search-filterable classification table.
2. **📁 Upload Reviews**: Drag-and-drop CSV uploader with dynamic column mapping and one-click benchmark dataset restore.
3. **🔍 Analyze Single Review**: Interactive test bench allowing users to input raw text, inspect step-by-step preprocessing transformations, and run instant anomaly scans.
4. **⛏️ Data Mining Deep Dive**:
   - **Tab 1: Classification**: Side-by-side benchmark of Naive Bayes, Logistic Regression, Random Forest, and SVM with confusion matrices.
   - **Tab 2: Clustering & PCA**: 2D scatter visualization of TF-IDF clusters with custom cluster themes.
   - **Tab 3: Frequency Mining**: Unigram, Bigram, and Trigram horizontal bar distributions.
   - **Tab 4: Association Rules**: Aspect-outcome rule table with interactive Support vs Confidence bubble scatter.
   - **Tab 5: Anomaly Audit**: Audit log of all flagged suspicious reviews with itemized reasons.
5. **📑 Reputation Report**: One-click generation and download of complete executive audit reports in Markdown.
6. **📚 Methodology & Architecture**: Built-in reference handbook explaining every mathematical formula and KDD stage.

---
_Developed as a complete demonstration of Data Mining and Text Mining principles._
