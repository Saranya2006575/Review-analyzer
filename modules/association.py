"""
Association Analysis Module
Implements Frequent Itemset Mining and Association Rule Mining to uncover
correlations between specific product/service aspects and sentiment/rating outcomes.
Calculates Support, Confidence, and Lift.
"""

import re
from itertools import combinations
from typing import Dict, List, Tuple, Any
import pandas as pd


# Aspect dictionary mapping keywords to core conceptual aspects
ASPECT_TAXONOMY = {
    "Battery & Charging": ["battery", "charge", "charging", "charger", "drain", "power", "discharges"],
    "Audio & Sound": ["sound", "audio", "bass", "treble", "acoustic", "mic", "microphone", "noise", "volume", "music"],
    "Customer Support": ["support", "service", "warranty", "refund", "ticket", "email", "representative", "bot", "credit"],
    "Build Quality": ["build", "cushion", "headband", "hinge", "strap", "plastic", "titanium", "durability", "scratched", "bezel"],
    "Software & App": ["app", "software", "sync", "bluetooth", "connection", "connect", "firmware", "bug", "glitch", "crash"],
    "Delivery & Packaging": ["delivery", "courier", "driver", "package", "packaging", "box", "shipping", "ontime", "late", "food", "spilled"],
    "Pricing & Value": ["price", "cost", "subscription", "worth", "tier", "fee", "fees", "expensive", "paywall"]
}


class AssociationMiner:
    """
    Data Mining Association Rule Engine.
    Discovers rules like: {Aspect: Battery & Charging} => {Sentiment: Negative}
    with metrics: Support, Confidence, and Lift.
    """

    def __init__(self, aspect_taxonomy: Dict[str, List[str]] = None):
        self.taxonomy = aspect_taxonomy or ASPECT_TAXONOMY

    def extract_aspects(self, text: str) -> List[str]:
        """Identifies which functional aspects are referenced in a given review."""
        lower_text = text.lower()
        detected = []
        for aspect, keywords in self.taxonomy.items():
            for kw in keywords:
                if re.search(r'\b' + re.escape(kw) + r'\b', lower_text):
                    detected.append(aspect)
                    break
        return detected

    def build_transactions(self, df: pd.DataFrame, text_col: str = 'Review Text') -> List[List[str]]:
        """
        Translates customer reviews into transactional baskets:
        Basket = [Aspect: X, Aspect: Y, Sentiment: Positive, Rating: High (4-5)]
        """
        transactions = []
        for _, row in df.iterrows():
            basket = []
            # 1. Aspects
            aspects = self.extract_aspects(str(row.get(text_col, '')))
            for asp in aspects:
                basket.append(f"Aspect: {asp}")

            # 2. Sentiment
            sent = row.get('Sentiment', None)
            if sent:
                basket.append(f"Sentiment: {sent}")

            # 3. Rating bracket
            rating = row.get('Rating', None)
            if rating is not None:
                try:
                    r_val = float(rating)
                    if r_val >= 4.0:
                        basket.append("Rating: High (4-5)")
                    elif r_val <= 2.0:
                        basket.append("Rating: Low (1-2)")
                    else:
                        basket.append("Rating: Mid (3)")
                except Exception:
                    pass

            transactions.append(basket)
        return transactions

    def mine_rules(
        self,
        df: pd.DataFrame,
        text_col: str = 'Review Text',
        min_support: float = 0.05,
        min_confidence: float = 0.30
    ) -> pd.DataFrame:
        """
        Mines meaningful association rules connecting Aspects to Sentiments / Ratings.
        Calculates Support, Confidence, and Lift.
        """
        transactions = self.build_transactions(df, text_col=text_col)
        N = len(transactions)
        if N == 0:
            return pd.DataFrame(columns=['Antecedent', 'Consequent', 'Support (%)', 'Confidence (%)', 'Lift'])

        # 1. Frequency count of single items
        item_counts = {}
        for t in transactions:
            for item in set(t):
                item_counts[item] = item_counts.get(item, 0) + 1

        # 2. Pair counts (Antecedent -> Consequent) where Antecedent is Aspect and Consequent is Sentiment/Rating
        pair_counts = {}
        for t in transactions:
            items = list(set(t))
            aspect_items = [i for i in items if i.startswith("Aspect:")]
            outcome_items = [i for i in items if i.startswith("Sentiment:") or i.startswith("Rating:")]

            for asp in aspect_items:
                for out in outcome_items:
                    pair = (asp, out)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

        # 3. Calculate metrics
        rules = []
        for (antecedent, consequent), count_ab in pair_counts.items():
            support = count_ab / N
            if support < min_support:
                continue

            count_a = item_counts.get(antecedent, 0)
            count_b = item_counts.get(consequent, 0)

            if count_a == 0 or count_b == 0:
                continue

            confidence = count_ab / count_a
            if confidence < min_confidence:
                continue

            expected_support = count_b / N
            lift = confidence / expected_support if expected_support > 0 else 0.0

            rules.append({
                "Antecedent": antecedent.replace("Aspect: ", ""),
                "Consequent": consequent,
                "Support (%)": round(support * 100, 2),
                "Confidence (%)": round(confidence * 100, 2),
                "Lift": round(lift, 2),
                "Co-occurrences": count_ab
            })

        if not rules:
            return pd.DataFrame(columns=['Antecedent', 'Consequent', 'Support (%)', 'Confidence (%)', 'Lift'])

        rules_df = pd.DataFrame(rules).sort_values(by=['Lift', 'Confidence (%)'], ascending=[False, False]).reset_index(drop=True)
        return rules_df
