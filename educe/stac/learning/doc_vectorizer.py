"""This submodule implements document vectorizers"""

# pylint: disable=too-few-public-methods

from .features import clean_dialogue_act


UNK = '__UNK__'
ROOT = 'ROOT'
UNRELATED = 'UNRELATED'


class DialogueActVectorizer(object):
    """Dialogue act extractor for the STAC corpus."""

    def __init__(self, instance_generator, labels):
        """
        instance_generator to enumerate the instances from a doc

        :type labels: set(string)
        """
        self.instance_generator = instance_generator
        self.labelset_ = {l: i for i, l in enumerate(labels, start=1)}
        self.labelset_[UNK] = 0

    def transform(self, raw_documents):
        """Learn the label encoder and return a vector of labels

        There is one label per instance extracted from raw_documents.
        """
        # run through documents to generate y
        for doc in raw_documents:
            for edu in self.instance_generator(doc):
                label = clean_dialogue_act(edu.dialogue_act() or UNK)
                yield self.labelset_[label]


class LabelVectorizer(object):
    """Label extractor for the STAC corpus."""

    def __init__(self, instance_generator, labels, zero=False):
        """
        instance_generator to enumerate the instances from a doc

        :type labels: set(string)
        """
        self.instance_generator = instance_generator
        self.labelset_ = {l: i for i, l in enumerate(labels, start=3)}
        self.labelset_[UNK] = 0
        self.labelset_[ROOT] = 1
        self.labelset_[UNRELATED] = 2
        self._zero = zero

    def transform(self, raw_documents):
        """Learn the label encoder and return a vector of labels

        There is one label per instance extracted from raw_documents.
        """
        zlabel = UNK if self._zero else UNRELATED
        # run through documents to generate y
        for doc in raw_documents:
            for pair in self.instance_generator(doc):
                label = doc.relations.get(pair, zlabel)
                yield self.labelset_[label]
