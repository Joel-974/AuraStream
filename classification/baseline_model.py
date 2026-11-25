"""
Pure-Python Baseline Text Classification Pipeline for AuraStream.
Implements TF-IDF vectorization and Multinomial Naive Bayes with Laplace smoothing.
100% dependency-free: no unsigned binary C-extensions, fully immune to Windows Smart App Control.
Saves human-readable model parameters and probability tables as clean JSON.
"""

import math
import re
import json
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# English stopwords list
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "did", "do", "does", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just",
    "me", "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once",
    "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same", "she",
    "should", "so", "some", "such", "t", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "you", "your", "yours", "yourself",
}


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase words, filtering short tokens and stopwords."""
    words = re.findall(r"\b[a-zA-Z]{2,}\b", text.lower())
    return [w for w in words if w not in STOP_WORDS]


class PureTfidfVectorizer:
    """TF-IDF Feature Extractor in Pure Python."""

    def __init__(self, max_features: int = 4000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocabulary: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.num_docs: int = 0

    def fit(self, documents: List[str]):
        """Fit vocabulary and compute inverse document frequency (IDF)."""
        self.num_docs = len(documents)
        doc_freq = Counter()

        for doc in documents:
            tokens = set(tokenize(doc))
            for token in tokens:
                doc_freq[token] += 1

        # Select top features meeting min_df
        valid_terms = [t for t, freq in doc_freq.items() if freq >= self.min_df]
        # Sort by document frequency descending
        valid_terms.sort(key=lambda t: doc_freq[t], reverse=True)
        top_terms = valid_terms[: self.max_features]

        self.vocabulary = {term: idx for idx, term in enumerate(top_terms)}
        # Smooth IDF: log((1 + N) / (1 + df)) + 1
        self.idf = {
            term: math.log((1 + self.num_docs) / (1 + doc_freq[term])) + 1.0
            for term in top_terms
        }

    def transform_single(self, text: str) -> Dict[str, float]:
        """Transform a single text string into a normalized TF-IDF sparse dictionary."""
        tokens = tokenize(text)
        if not tokens:
            return {}

        term_counts = Counter(t for t in tokens if t in self.vocabulary)
        if not term_counts:
            return {}

        # Sublinear term frequency: 1 + log(tf)
        tfidf_vec = {}
        for term, count in term_counts.items():
            tf = 1.0 + math.log(count)
            tfidf_vec[term] = tf * self.idf[term]

        # L2 Normalization
        norm = math.sqrt(sum(v * v for v in tfidf_vec.values()))
        if norm > 0:
            for term in tfidf_vec:
                tfidf_vec[term] /= norm

        return tfidf_vec


class PureNaiveBayesClassifier:
    """Multinomial Naive Bayes Classifier with Laplace Smoothing and Balanced Class Priors."""

    def __init__(self, alpha: float = 1.0, balanced_priors: bool = True):
        self.alpha = alpha
        self.balanced_priors = balanced_priors
        self.classes: List[str] = []
        self.class_log_priors: Dict[str, float] = {}
        self.feature_log_probs: Dict[str, Dict[str, float]] = {}
        self.default_feature_log_probs: Dict[str, float] = {}
        self.vectorizer: Optional[PureTfidfVectorizer] = None

    def fit(self, documents: List[str], labels: List[str], max_features: int = 4000):
        """Train TF-IDF vectorizer and compute likelihood distributions."""
        self.classes = sorted(list(set(labels)))
        self.vectorizer = PureTfidfVectorizer(max_features=max_features)
        self.vectorizer.fit(documents)

        # Class counts
        class_counts = Counter(labels)
        total_docs = len(labels)
        num_classes = len(self.classes)

        # Class Priors
        if self.balanced_priors:
            # Uniform / balanced prior gives rare critical classes fair probability
            for cls_name in self.classes:
                self.class_log_priors[cls_name] = math.log(1.0 / num_classes)
        else:
            for cls_name in self.classes:
                self.class_log_priors[cls_name] = math.log(class_counts[cls_name] / total_docs)

        # Aggregate feature weights per class
        class_word_weights = {cls_name: defaultdict(float) for cls_name in self.classes}
        class_total_weights = {cls_name: 0.0 for cls_name in self.classes}

        for doc, label in zip(documents, labels):
            tfidf = self.vectorizer.transform_single(doc)
            for term, weight in tfidf.items():
                class_word_weights[label][term] += weight
                class_total_weights[label] += weight

        vocab_size = len(self.vectorizer.vocabulary)
        self.feature_log_probs = {cls_name: {} for cls_name in self.classes}
        self.default_feature_log_probs = {}

        for cls_name in self.classes:
            denominator = class_total_weights[cls_name] + (self.alpha * vocab_size)
            # Precompute log-probability for unseen tokens
            self.default_feature_log_probs[cls_name] = math.log(self.alpha / denominator)

            for term in self.vectorizer.vocabulary:
                numerator = class_word_weights[cls_name].get(term, 0.0) + self.alpha
                self.feature_log_probs[cls_name][term] = math.log(numerator / denominator)

    def predict_proba_single(self, text: str) -> Dict[str, float]:
        """Compute posterior probabilities over all classes using softmax over log-posteriors."""
        if not self.vectorizer:
            raise RuntimeError("Model has not been fitted.")

        tfidf = self.vectorizer.transform_single(text)
        if not tfidf:
            # Return uniform probability if document contains no recognized vocabulary
            p = 1.0 / len(self.classes)
            return {cls_name: round(p, 4) for cls_name in self.classes}

        # Calculate log posteriors: log P(c) + sum(log P(w|c) * weight)
        log_posteriors = {}
        for cls_name in self.classes:
            log_prob = self.class_log_priors[cls_name]
            feature_probs = self.feature_log_probs[cls_name]
            default_prob = self.default_feature_log_probs[cls_name]

            for term, weight in tfidf.items():
                word_log_prob = feature_probs.get(term, default_prob)
                log_prob += word_log_prob * (1.0 + weight)

            log_posteriors[cls_name] = log_prob

        # Numerically stable softmax: exp(x - max_x) / sum(exp(x - max_x))
        max_log = max(log_posteriors.values())
        exp_posteriors = {cls_name: math.exp(val - max_log) for cls_name, val in log_posteriors.items()}
        sum_exp = sum(exp_posteriors.values())

        return {
            cls_name: round(exp_val / sum_exp, 4)
            for cls_name, exp_val in exp_posteriors.items()
        }

    def predict_single(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict top class, confidence score, and full distribution."""
        probs = self.predict_proba_single(text)
        top_class = max(probs, key=probs.get)
        top_conf = probs[top_class]
        return top_class, top_conf, probs

    def save(self, filepath: Path):
        """Save trained model parameters and vocabulary to JSON."""
        state = {
            "classes": self.classes,
            "alpha": self.alpha,
            "balanced_priors": self.balanced_priors,
            "class_log_priors": self.class_log_priors,
            "default_feature_log_probs": self.default_feature_log_probs,
            "feature_log_probs": self.feature_log_probs,
            "vocabulary": self.vectorizer.vocabulary,
            "idf": self.vectorizer.idf,
            "num_docs": self.vectorizer.num_docs,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(state, f)

    @classmethod
    def load(cls, filepath: Path) -> "PureNaiveBayesClassifier":
        """Load trained model from JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)

        model = cls(alpha=state["alpha"], balanced_priors=state["balanced_priors"])
        model.classes = state["classes"]
        model.class_log_priors = state["class_log_priors"]
        model.default_feature_log_probs = state["default_feature_log_probs"]
        model.feature_log_probs = state["feature_log_probs"]

        vec = PureTfidfVectorizer()
        vec.vocabulary = state["vocabulary"]
        vec.idf = state["idf"]
        vec.num_docs = state["num_docs"]
        model.vectorizer = vec

        return model
