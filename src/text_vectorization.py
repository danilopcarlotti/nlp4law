from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


from .text_normalization import normalize_texts


def tfidf_texts(texts, ngram_range=(1, 1)):
    corpus = normalize_texts(texts)
    return TfidfVectorizer(ngram_range=ngram_range).fit_transform(corpus).toarray()


def hashing_texts(texts, n_features, ngram_range=(1, 1)):
    corpus = normalize_texts(texts)
    return (
        HashingVectorizer(n_features=n_features, ngram_range=ngram_range)
        .fit_transform(corpus)
        .toarray()
    )
