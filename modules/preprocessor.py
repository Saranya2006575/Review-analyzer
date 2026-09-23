"""
Preprocessor Module
Handles text cleaning, tokenization, stopword removal, lemmatization,
duplicate detection, and missing value handling.
"""

import re
import string
import pandas as pd
from typing import Dict, List, Tuple, Any

# Common English stopwords fallback list to ensure zero-failure execution
FALLBACK_STOPWORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any',
    'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
    'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
    'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
    'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't",
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma',
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't",
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't",
    'wouldn', "wouldn't"
}

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.tokenize import word_tokenize

    # Check local NLTK datasets without blocking network calls
    try:
        nltk.data.find('corpora/stopwords')
        STOP_WORDS = set(stopwords.words('english'))
    except Exception:
        STOP_WORDS = FALLBACK_STOPWORDS

    STEMMER = PorterStemmer()

    try:
        nltk.data.find('corpora/wordnet')
        LEMMATIZER = WordNetLemmatizer()
    except Exception:
        LEMMATIZER = None

    NLTK_AVAILABLE = True
except Exception:
    STOP_WORDS = FALLBACK_STOPWORDS
    LEMMATIZER = None
    STEMMER = None
    NLTK_AVAILABLE = False


class TextPreprocessor:
    """
    Data Mining Text Preprocessing Pipeline.
    Supports clean tokenization, lemmatization, stop words removal,
    missing value imputation, and duplicate detection.
    """

    def __init__(self, use_stemming: bool = False):
        self.use_stemming = use_stemming
        self.stop_words = STOP_WORDS

    def clean_raw_text(self, text: str) -> str:
        """
        Applies regex cleaning:
        - Casts to string
        - Removes HTML tags
        - Removes URLs
        - Converts to lowercase
        - Strips punctuation and digits
        - Collapses excess whitespace
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Remove HTML
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Remove emails
        text = re.sub(r'\S+@\S+', ' ', text)
        # Remove punctuation and special characters (keep only letters and spaces)
        text = re.sub(r'[^a-z\s]', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenizes text into words.
        Uses regex or NLTK punkt.
        """
        cleaned = self.clean_raw_text(text)
        if not cleaned:
            return []
        if NLTK_AVAILABLE and LEMMATIZER is not None:
            try:
                tokens = word_tokenize(cleaned)
            except Exception:
                tokens = re.findall(r'\b[a-z]{2,}\b', cleaned)
        else:
            tokens = re.findall(r'\b[a-z]{2,}\b', cleaned)
        return tokens

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Removes common stop words from token list."""
        return [tok for tok in tokens if tok not in self.stop_words and len(tok) > 1]

    def lemmatize_or_stem(self, tokens: List[str]) -> List[str]:
        """Performs Lemmatization or Stemming on words."""
        if self.use_stemming and STEMMER:
            return [STEMMER.stem(tok) for tok in tokens]
        elif LEMMATIZER:
            try:
                return [LEMMATIZER.lemmatize(tok) for tok in tokens]
            except Exception:
                return tokens
        return tokens

    def preprocess_string(self, text: str) -> str:
        """Complete transformation pipeline returning clean joined text."""
        tokens = self.tokenize(text)
        filtered = self.remove_stopwords(tokens)
        lemmatized = self.lemmatize_or_stem(filtered)
        return ' '.join(lemmatized)

    def get_step_by_step(self, text: str) -> Dict[str, Any]:
        """
        Generates step-by-step transformation traces for educational/inspection purposes.
        """
        step1_raw = str(text)
        step2_cleaned = self.clean_raw_text(step1_raw)
        step3_tokens = self.tokenize(step1_raw)
        step4_no_stopwords = self.remove_stopwords(step3_tokens)
        step5_lemmatized = self.lemmatize_or_stem(step4_no_stopwords)
        step6_final = ' '.join(step5_lemmatized)

        return {
            "Original Text": step1_raw,
            "Cleaned Lowercase": step2_cleaned,
            "Tokens": step3_tokens,
            "Stopwords Removed": step4_no_stopwords,
            "Lemmatized Tokens": step5_lemmatized,
            "Final Clean Text": step6_final,
            "Original Word Count": len(step1_raw.split()),
            "Processed Word Count": len(step5_lemmatized)
        }

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = 'Review Text',
        rating_col: str = 'Rating',
        dedup_exact: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Preprocesses an entire reviews DataFrame:
        - Logs initial count
        - Drops or imputes missing values
        - Detects and removes exact duplicate reviews
        - Normalizes ratings to float/int
        - Generates 'Clean_Review' column
        - Returns preprocessed DataFrame and metadata summary
        """
        initial_rows = len(df)
        df_clean = df.copy()

        # 1. Missing value handling
        nulls_in_text = df_clean[text_col].isnull().sum()
        df_clean = df_clean.dropna(subset=[text_col])
        df_clean[text_col] = df_clean[text_col].astype(str)

        if rating_col in df_clean.columns:
            nulls_in_rating = df_clean[rating_col].isnull().sum()
            # Impute missing rating with median
            median_rating = df_clean[rating_col].median() if len(df_clean) > 0 else 3.0
            df_clean[rating_col] = pd.to_numeric(df_clean[rating_col], errors='coerce').fillna(median_rating)
        else:
            nulls_in_rating = 0

        # 2. Duplicate detection & removal
        duplicates_removed = 0
        if dedup_exact:
            # Check duplicate on text_col
            dups_mask = df_clean.duplicated(subset=[text_col], keep='first')
            duplicates_removed = int(dups_mask.sum())
            df_clean = df_clean[~dups_mask].copy()

        # 3. Text cleaning pipeline
        df_clean['Clean_Review'] = df_clean[text_col].apply(self.preprocess_string)
        # Drop empty reviews after cleaning
        df_clean = df_clean[df_clean['Clean_Review'].str.strip() != '']

        # Word count metrics
        df_clean['Word_Count'] = df_clean[text_col].apply(lambda x: len(str(x).split()))
        df_clean['Clean_Word_Count'] = df_clean['Clean_Review'].apply(lambda x: len(str(x).split()))

        summary = {
            "initial_rows": initial_rows,
            "final_rows": len(df_clean),
            "nulls_in_text": int(nulls_in_text),
            "nulls_in_rating": int(nulls_in_rating),
            "duplicates_removed": duplicates_removed,
            "vocab_reduction_pct": round(
                (1 - (df_clean['Clean_Word_Count'].sum() / max(1, df_clean['Word_Count'].sum()))) * 100, 2
            )
        }

        return df_clean.reset_index(drop=True), summary
