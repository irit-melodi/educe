"""This module provides ways to transform lists of PairKeys to sparse vectors.
"""

from itertools import chain
from collections import defaultdict


class PairKeysVectorizer(object):
    """Transforms lists of PairKeys to sparse vectors.
    """

    def _count_vocab(self, pair_keys, fixed_vocab):
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
        pair_ptr = []
        pair_ptr.append(0)

        for vec in pair_keys:
            for feature, featval in vec.one_hot_values_gen():
                try:
                    feature_acc.append((vocabulary[feature], featval))
                except KeyError:
                    # ignore unknown features if fixed vocab
                    continue
            pair_ptr.append(len(feature_acc))

        if not fixed_vocab:
            vocabulary = dict(vocabulary)
            if not vocabulary:
                raise ValueError("empty vocabulary")

        # build a feature count matrix out of feature_acc and pair_ptr
        X = []
        for i in xrange(len(pair_ptr) - 1):
            current_pair, next_pair = pair_ptr[i], pair_ptr[i + 1]
            x = feature_acc[current_pair:next_pair]
            X.append(x)
        return vocabulary, X

    def fit_transform(self, edu_pairs, y=None):
        """Learn the vocabulary dictionary and return instances
        """
        vocabulary, X = self._count_vocab(edu_pairs, fixed_vocab=False)
        self.vocabulary_ = vocabulary
        return X

    def transform(self, edu_pairs):
        """Transform documents to EDU pair feature matrix.

        Extract features out of documents using the vocabulary
        fitted with fit.
        """
        _, X = self._count_vocab(edu_pairs, fixed_vocab=True)
        return X


class StreamingPairKeysVectorizer(object):
    """Transforms lists of PairKeys to sparse vectors, streaming version.
    """

    def _count_vocab(self, pair_keys, fixed_vocab):
        """Create sparse feature matrix and vocabulary
        """
        if fixed_vocab:
            vocabulary = self.vocabulary_
            classes = self.classes_
        else:
            # every time a new value is encountered, add it to the vocabulary
            vocabulary = defaultdict()
            vocabulary.default_factory = vocabulary.__len__
            # same for labels
            classes = defaultdict()
            classes.default_factory = classes.__len__

        for vec, label in pair_keys:
            feature_acc = []
            for feature, featval in vec.one_hot_values_gen():
                try:
                    feature_acc.append((vocabulary[feature], featval))
                except KeyError:
                    # ignore unknown features if fixed vocab
                    continue
            try:
                label = classes[label]
            except KeyError:
                # TODO: check is -1 is a plausible value for unknown labels
                # with fixed vocabulary
                label = -1
            yield vocabulary, classes, feature_acc, label

    def fit_transform(self, edu_pairs, y=None):
        """Learn the vocabulary dictionary and return instances
        """
        for vocab, classes, x, yi in self._count_vocab(edu_pairs,
                                                       fixed_vocab=False):
            yield x, yi
        # freeze and set the vocabulary
        vocabulary = dict(vocab)
        if not vocabulary:
            raise ValueError("empty vocabulary")
        self.vocabulary_ = vocabulary
        # freeze and set the labels
        classes = dict(classes)
        self.classes_ = classes

    def transform(self, edu_pairs):
        """Transform documents to EDU pair feature matrix.

        Extract features out of documents using the vocabulary
        fitted with fit.
        """
        for _, _, x, yi in self._count_vocab(edu_pairs, fixed_vocab=True):
            yield x, yi
