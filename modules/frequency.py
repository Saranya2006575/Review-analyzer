"""
Frequency Analysis Module
Performs N-gram word frequency mining, positive/negative keyword extraction,
and praised features vs customer complaints detection.
"""

import re
from collections import Counter
from typing import Dict, List, Tuple, Any
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer


class FrequencyMiner:
    """
    Data Mining frequency analyzer for n-grams and aspect extraction.
    """

    def __init__(self, stop_words: str = 'english'):
        self.stop_words = stop_words

    def get_top_ngrams(
        self,
        texts: List[str],
        ngram_range: Tuple[int, int] = (1, 1),
        top_n: int = 15
    ) -> pd.DataFrame:
        """
        Extracts top N-grams from text corpus.
        """
        valid_texts = [str(t) for t in texts if str(t).strip()]
        if not valid_texts:
            return pd.DataFrame(columns=['Ngram', 'Frequency'])

        try:
            vec = CountVectorizer(ngram_range=ngram_range, stop_words=self.stop_words, min_df=1)
            X = vec.fit_transform(valid_texts)
            sum_words = X.sum(axis=0)
            words_freq = [(word, int(sum_words[0, idx])) for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)[:top_n]
            return pd.DataFrame(words_freq, columns=['Ngram', 'Frequency'])
        except Exception:
            # Fallback simple counter
            words = []
            for t in valid_texts:
                toks = re.findall(r'\b[a-z]{3,}\b', t.lower())
                words.extend(toks)
            c = Counter(words).most_common(top_n)
            return pd.DataFrame(c, columns=['Ngram', 'Frequency'])

    def analyze_sentiment_keywords(
        self,
        df: pd.DataFrame,
        text_col: str = 'Clean_Review',
        sentiment_col: str = 'Sentiment',
        top_n: int = 12
    ) -> Dict[str, pd.DataFrame]:
        """
        Extracts distinctive positive keywords, negative keywords,
        and frequent bigram complaints/praise.
        """
        pos_texts = df[df[sentiment_col] == 'Positive'][text_col].tolist()
        neg_texts = df[df[sentiment_col] == 'Negative'][text_col].tolist()

        pos_unigrams = self.get_top_ngrams(pos_texts, ngram_range=(1, 1), top_n=top_n)
        neg_unigrams = self.get_top_ngrams(neg_texts, ngram_range=(1, 1), top_n=top_n)

        pos_bigrams = self.get_top_ngrams(pos_texts, ngram_range=(2, 2), top_n=top_n)
        neg_bigrams = self.get_top_ngrams(neg_texts, ngram_range=(2, 2), top_n=top_n)

        return {
            "positive_keywords": pos_unigrams,
            "negative_keywords": neg_unigrams,
            "praised_aspects": pos_bigrams,
            "common_complaints": neg_bigrams
        }
