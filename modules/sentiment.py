"""
Sentiment Analysis Module
Analyzes review polarity, computes normalized sentiment scores (-1.0 to 1.0),
and classifies text into Positive, Neutral, or Negative.
Features both VADER sentiment analysis and a robust built-in rule-based lexicon fallback.
"""

import re
import pandas as pd
from typing import Dict, Any, Tuple

# Built-in robust polarity lexicon dictionary for offline/instant execution
BUILTIN_LEXICON = {
    # Strong positive (+2.0 to +3.0)
    'phenomenal': 3.0, 'exceptional': 3.0, 'breathtaking': 3.0, 'outstanding': 3.0,
    'flawless': 3.0, 'superb': 2.5, 'magnificent': 2.5, 'stellar': 2.5,
    'luxurious': 2.5, 'masterpiece': 3.0, 'unbelievable': 2.0, 'exceeded': 2.5,
    'transcends': 2.5, 'lifesaver': 2.5, 'impeccable': 3.0, 'delightful': 2.5,
    # Standard positive (+1.0 to +2.0)
    'great': 1.8, 'excellent': 2.2, 'good': 1.2, 'love': 2.0, 'loved': 2.0,
    'best': 2.2, 'awesome': 2.0, 'top': 1.5, 'fast': 1.3, 'rapid': 1.2,
    'clean': 1.2, 'sleek': 1.5, 'comfortable': 1.8, 'reliable': 1.8,
    'crisp': 1.4, 'punchy': 1.4, 'satisfied': 1.5, 'satisfaction': 1.5,
    'solid': 1.3, 'sturdy': 1.4, 'rich': 1.4, 'convenient': 1.5,
    'safe': 1.4, 'dependable': 1.6, 'swift': 1.5, 'helpful': 1.4,
    'worth': 1.5, 'smooth': 1.4, 'easy': 1.3, 'seamless': 1.8,
    'accurate': 1.6, 'impressive': 1.8, 'pleased': 1.5, 'recommend': 1.8,
    'clear': 1.3, 'clarity': 1.4, 'perfect': 2.2, 'bright': 1.2,
    # Weak positive (+0.5 to +0.8)
    'decent': 0.6, 'acceptable': 0.5, 'okay': 0.4, 'ok': 0.3, 'fine': 0.5,
    # Weak negative (-0.5 to -0.8)
    'mediocre': -0.7, 'slow': -0.8, 'weak': -0.8, 'noisy': -0.7,
    'sweaty': -0.6, 'stiff': -0.6, 'bulky': -0.5, 'sluggish': -0.8,
    # Standard negative (-1.0 to -2.0)
    'bad': -1.8, 'poor': -1.8, 'broken': -2.0, 'broke': -2.0,
    'fail': -1.8, 'fails': -1.8, 'failed': -1.8, 'glitch': -1.5,
    'bug': -1.4, 'bugs': -1.5, 'annoying': -1.5, 'disappointed': -1.8,
    'disappointing': -1.8, 'drop': -1.2, 'drops': -1.2, 'latency': -1.3,
    'refused': -1.8, 'hiss': -1.4, 'drain': -1.5, 'freeze': -1.5,
    'freezes': -1.5, 'crash': -2.0, 'crashes': -2.0, 'charge': -0.5,
    'corrupted': -2.2, 'shattered': -2.2, 'brittle': -1.8, 'cheap': -1.6,
    'headache': -1.8, 'pain': -1.6, 'rash': -2.0, 'allergic': -1.8,
    'rude': -2.0, 'stole': -2.5, 'spilled': -1.6, 'soggy': -1.8,
    'lukewarm': -1.2, 'cold': -1.2, 'exorbitant': -1.8, 'inflated': -1.5,
    # Strong negative (-2.5 to -3.0)
    'terrible': -2.8, 'awful': -2.8, 'horrible': -2.8, 'worst': -3.0,
    'hate': -2.5, 'hated': -2.5, 'disaster': -2.8, 'disgraceful': -2.8,
    'disgusting': -2.8, 'scam': -3.0, 'unusable': -2.6, 'defective': -2.5,
    'bricked': -2.8, 'useless': -2.6, 'unacceptable': -2.8, 'trash': -2.8
}

NEGATION_WORDS = {'not', 'no', 'never', 'neither', 'hardly', 'barely', 'scarcely', "n't", "cannot", "without"}
INTENSIFIERS = {'very': 1.5, 'extremely': 1.8, 'super': 1.6, 'incredibly': 1.8, 'really': 1.4, 'deeply': 1.4, 'completely': 1.5, 'totally': 1.5}


class SentimentAnalyzer:
    """
    Sentiment Analyzer providing hybrid Lexicon and ML sentiment scoring.
    """

    def __init__(self):
        self.vader = None
        try:
            import nltk
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            nltk.data.find('sentiment/vader_lexicon.zip')
            self.vader = SentimentIntensityAnalyzer()
        except Exception:
            self.vader = None

    def analyze_vader(self, text: str) -> Dict[str, float]:
        """Runs NLTK VADER sentiment analyzer if available."""
        if self.vader:
            try:
                return self.vader.polarity_scores(text)
            except Exception:
                pass
        return {}

    def analyze_lexicon(self, text: str) -> Tuple[float, str]:
        """
        Rule-based lexicon analyzer supporting negations and intensifiers.
        Computes score in range [-1.0, 1.0].
        """
        words = re.findall(r'\b[a-zA-Z\']+\b', text.lower())
        if not words:
            return 0.0, "Neutral"

        total_score = 0.0
        counted_words = 0
        i = 0
        n = len(words)

        while i < n:
            word = words[i]
            # Check negation in window of 2 previous words
            negated = False
            for j in range(max(0, i - 2), i):
                if words[j] in NEGATION_WORDS:
                    negated = True
                    break

            # Check intensifier immediately prior
            multiplier = 1.0
            if i > 0 and words[i - 1] in INTENSIFIERS:
                multiplier = INTENSIFIERS[words[i - 1]]

            if word in BUILTIN_LEXICON:
                val = BUILTIN_LEXICON[word] * multiplier
                if negated:
                    val = -val * 0.8
                total_score += val
                counted_words += 1

            i += 1

        if counted_words == 0:
            score = 0.0
        else:
            # Normalized bounded between -1.0 and +1.0 using hyperbolic tangent / scaling
            raw_avg = total_score / (counted_words ** 0.5)
            # Clip between -1 and 1
            score = max(-1.0, min(1.0, raw_avg / 3.0))

        if score >= 0.05:
            label = "Positive"
        elif score <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        return round(score, 4), label

    def analyze_review(self, text: str) -> Dict[str, Any]:
        """
        Computes sentiment metrics for a single review string.
        """
        raw_text = str(text) if text is not None else ""

        # Check VADER first
        vader_scores = self.analyze_vader(raw_text)
        if vader_scores and 'compound' in vader_scores:
            compound = vader_scores['compound']
            if compound >= 0.05:
                sentiment = "Positive"
            elif compound <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"

            pos_prob = vader_scores.get('pos', 0.0)
            neg_prob = vader_scores.get('neg', 0.0)
            neu_prob = vader_scores.get('neu', 0.0)
        else:
            compound, sentiment = self.analyze_lexicon(raw_text)
            if sentiment == "Positive":
                pos_prob, neg_prob, neu_prob = round(0.5 + abs(compound) * 0.5, 3), 0.05, 0.1
            elif sentiment == "Negative":
                pos_prob, neg_prob, neu_prob = 0.05, round(0.5 + abs(compound) * 0.5, 3), 0.1
            else:
                pos_prob, neg_prob, neu_prob = 0.2, 0.2, 0.6

        return {
            "sentiment": sentiment,
            "sentiment_score": round(float(compound), 3),
            "pos_prob": float(pos_prob),
            "neu_prob": float(neu_prob),
            "neg_prob": float(neg_prob)
        }

    def add_sentiment_columns(self, df: pd.DataFrame, text_col: str = 'Review Text') -> pd.DataFrame:
        """
        Enriches a DataFrame with sentiment classification and sentiment score.
        """
        df_res = df.copy()
        sentiments = []
        scores = []

        for txt in df_res[text_col]:
            res = self.analyze_review(txt)
            sentiments.append(res['sentiment'])
            scores.append(res['sentiment_score'])

        df_res['Sentiment'] = sentiments
        df_res['Sentiment_Score'] = scores
        return df_res
