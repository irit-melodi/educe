"""Somewhat higher level representation of STAC documents
than the usual Glozz layer.

Note that this is a relatively recent addition to Educe.
Up to the time of this writing (2015-03), we had two options
for dealing with STAC:

    * manually manipulating glozz objects via educe.annotation
    * dealing with some high-level but not particularly helpful
      hypergraph objects

We try to provide an intermediary in this layer by merging information
from several layers in one place.

A typical example might be to print a listing of

(edu1_id, edu2_id, edu1_dialogue_act, edu2_dialogue_act, relation_label)

This has always been a bit awkward when dealing with Glozz, because there
are separate annotations in different Glozz documents, the dialogue
acts in the 'units' stage; and the linked units in the discourse stage.
Combining these streams has always involved a certain amount of manual lookup,
which we hope to avoid with this fusion layer.

At the time of this writing, this will have a bit of emphasis on
feature-extraction
"""

# pylint: disable=too-few-public-methods

from __future__ import print_function
import itertools as itr

ROOT = 'ROOT'
"distinguished fake EDU id for machine learning applications"


class Dialogue(object):
    """STAC Dialogue

    Note that input EDUs should be sorted by span
    """
    def __init__(self, anno, edus, relations):
        self.edus = [FakeRootEDU()] + edus
        self.grouping = anno.identifier()
        # we start from 1 because 0 is for the fake root
        self.edu2sent = {i: e.subgrouping()
                         for i, e in enumerate(edus, start=1)}
        self.relations = relations

    def edu_pairs(self):
        """Return all EDU pairs within this dialogue.

        NB: this is a generator
        """
        i_edus = list(enumerate(self.edus))
        i_edus = i_edus[1:]  # drop left padding EDU
        for num1, edu1 in i_edus:
            # pylint: disable=cell-var-from-loop
            is_before = lambda x: x[0] <= num1
            # pylint: enable=cell-var-from-loop
            for _, edu2 in itr.dropwhile(is_before, i_edus):
                yield (edu1, edu2)
                yield (edu2, edu1)


class EDU(object):
    """STAC EDU

    If you can't find what you need here, try going down a level
    and using stac.annotation on the :pyclass:`educe.annotation.Unit`
    annotations instead)"""
    def __init__(self, doc, context,
                 discourse_anno,
                 unit_anno):
        self._doc = doc
        self._context = context
        self._anno = discourse_anno
        self._unit_anno = unit_anno
        self.span = self._anno.text_span()  # used by vectorizer

    def dialogue_act(self):
        """
        The (normalised) speech act associated with this EDU
        (None if unknown)
        """
        if self._unit_anno is None:
            return None
        else:
            dact = self._unit_anno.type
        if dact == 'Strategic_comment':
            return 'Other'
        elif dact == 'Segment':
            return None
        else:
            return dact

    def text(self):
        "The text for just this EDU"
        return self._doc.text(self.span)

    # pylint: disable=no-self-use
    def is_left_padding(self):
        "If this is a virtual EDU used in machine learning tasks"
        return False
    # pylint: enable=no-self-use

    def identifier(self):
        """Some kind of identifier string that uniquely identfies the EDU in
        the corpus. Because these are higher level annotations than in the
        Glozz layer we will use the 'local' identifier, which should be the
        same across stages"""
        return self._anno.identifier()

    def subgrouping(self):
        """What abstract subgrouping the EDU is in

        :rtype int
        """
        return self._context.turn.identifier()


# pylint: disable=no-self-use
class FakeRootEDU(object):
    """Virtual EDU to represent the notion of a fake root node
    sometimes used in dependency parsing applications
    """
    def is_left_padding(self):
        "If this is a virtual EDU used in machine learning tasks"
        return True

    def identifier(self):
        """Some kind of identifier string that uniquely identfies the EDU in
        the corpus. Because these are higher level annotations than in the
        Glozz layer we will use the 'local' identifier, which should be the
        same across stages"""
        return ROOT
# pylint: enable=no-self-use
