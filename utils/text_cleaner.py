from sklearn.base import BaseEstimator,TransformerMixin
import re
import emoji
import contractions

class TextCleanerNB(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_clean = X.copy()
        X_clean["text"] = X_clean["text"].apply(self._clean_text)

        return X_clean["text"]

    def _clean_text(self, text) -> str:
        # Handle NaN / non-string safely
        if not isinstance(text, str):
            text = ""

        # 1. Normalize apostrophes
        text = text.replace("’", "'").replace("`", "'")

        # 2. Expand contractions (VERY important for NB)
        text = contractions.fix(text)

        # 3. Lowercase
        text = text.lower()

        # 4. Remove URLs and emails
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)

        # 5. Convert emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))

        # 6. Remove punctuation & special characters
        # NB works better with pure word counts
        text = re.sub(r"[^\w\s]", " ", text)

        # 7. Remove digits (NB assumption-friendly)
        text = re.sub(r"\d+", " ", text)

        # 8. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 9. SAFETY: NB + CountVectorizer must not see empty docs
        if not text:
            text = "__empty__"

        return text
    


class TextCleanerTFIDF(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_clean = X.copy()
        X_clean["text"] = X_clean["text"].apply(self._clean_text)
        return X_clean["text"]

    def _clean_text(self, text) -> str:
        # 0. Handle non-string / NaN safely
        if not isinstance(text, str):
            text = ""

        # 1. Normalize apostrophes
        text = text.replace("’", "'").replace("`", "'")

        # 2. Lowercase
        text = text.lower()

        # 3. Expand contractions (important for TF-IDF + LR)
        text = contractions.fix(text)

        # 4. Remove URLs and emails
        text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)

        # 5. Convert emojis to text (retain sentiment)
        text = emoji.demojize(text, delimiters=(" ", " "))

        # 6. Normalize excessive punctuation (!!! ???)
        text = re.sub(r"[!?]{2,}", " ! ", text)

        # 7. Remove unwanted symbols
        # Keep: letters, numbers, spaces, !
        text = re.sub(r"[^a-z0-9\s!]", " ", text)

        # 8. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 9. CRITICAL SAFETY: never allow empty documents
        if not text:
            text = "__empty__"

        return text