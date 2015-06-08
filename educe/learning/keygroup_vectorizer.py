"""This module provides ways to transform lists of PairKeys to sparse vectors.
"""

# pylint: disable=invalid-name
# lots of scikit-conventional names here

from collections import defaultdict


class KeyGroupVectorizer(object):
    """Transforms lists of KeyGroups to sparse vectors.
    """
    def __init__(self):
        self.vocabulary_ = None

    def _count_vocab(self, vectors, fixed_vocab):
        """Create sparse feature matrix and vocabulary
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
        else:
            # every time a new value is encountered, add it to the vocabulary
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__

        # accumulate features from every vec
        feature_acc = []
        # thus we need to remember where each pair of EDUs and each document
        # begins
        row_ptr = []
        row_ptr.append(0)

        for vec in vectors:
            for feature, featval in vec.one_hot_values_gen():
                try:
                    feature_acc.append((vocabulary[feature], featval))
                except KeyError:
                    # ignore unknown features if fixed vocab
                    continue
            row_ptr.append(len(feature_acc))

        if not fixed_vocab:
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary")

        # build a feature count matrix out of feature_acc and row_ptr
        X = []
        for i in xrange(len(row_ptr) - 1):
            current_row, next_row = row_ptr[i], row_ptr[i + 1]
            x = feature_acc[current_row:next_row]
            X.append(x)
        return vocabulary, X

    def fit_transform(self, vectors):
        """Learn the vocabulary dictionary and return instances
        """
        vocabulary, X = self._count_vocab(vectors, fixed_vocab=False)
        self.vocabulary_ = vocabulary
        return X

    def transform(self, vectors):
        """Transform documents to EDU pair feature matrix.

        Extract features out of documents using the vocabulary
        fitted with fit.
        """
        _, X = self._count_vocab(vectors, fixed_vocab=True)
        return X
