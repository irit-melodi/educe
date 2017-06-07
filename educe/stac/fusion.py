"""Somewhat higher level representation of STAC documents than the usual
Glozz layer.

Note that this is a relatively recent addition to Educe.
Up to the time of this writing (2015-03), we had two options for dealing
with STAC:

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
Combining these streams has always involved a certain amount of manual
lookup, which we hope to avoid with this fusion layer.

At the time of this writing, this will have a bit of emphasis on feature
extraction.
"""

# pylint: disable=too-few-public-methods

from __future__ import print_function
import copy
import itertools

from educe.annotation import (Span, Unit)
from educe.stac.annotation import (is_edu, speaker, turn_id, twin_from)
from educe.stac.context import (Context)

ROOT = 'ROOT'
"distinguished fake EDU id for machine learning applications"


class Dialogue(object):
    """STAC Dialogue.

    Note that input EDUs should be sorted by span.

    """
    def __init__(self, anno, edus, relations):
        """
        Parameters
        ----------
        anno : educe.stac.annotation.Unit
            Glozz annotation corresponding to the dialogue ; only its
            identifier is stored, currently.
        edus : list of educe.stac.annotation.Unit
            List of EDU annotations, sorted by their span.
        relations : list of educe.stac.annotation.Relation
            List of relations between EDUs from the dialogue.
        """
        self.grouping = anno.identifier()
        self.edus = [FakeRootEDU] + edus
        self.relations = relations
        # we start from 1 because 0 is for the fake root
        self.edu2sent = {i: e.subgrouping()
                         for i, e in enumerate(edus, start=1)}

    def edu_pairs(self):
        """Generate all EDU pairs within this dialogue.

        This includes pairs whose source is the left padding (fake root)
        EDU.

        Yields
        ------
        (source, target) : tuple of educe.stac.annotation.Unit
            Next candidate edge, as a pair of EDUs (source, target).
        """
        i_edus = list(enumerate(self.edus))
        _, fakeroot = i_edus[0]
        i_edus = i_edus[1:]  # drop left padding EDU
        for _, edu in i_edus:
            yield (fakeroot, edu)
        # generate all pairs of (real) EDUs
        for num1, edu1 in i_edus:
            def is_before(numedu2):
                'true if we have seen the EDU already'
                # pylint: disable=cell-var-from-loop
                num2 = numedu2[0]
                return num2 <= num1
                # pylint: enable=cell-var-from-loop
            for _, edu2 in itertools.dropwhile(is_before, i_edus):
                yield (edu1, edu2)
                yield (edu2, edu1)


# pylint: disable=too-many-instance-attributes
# we're trying to cover a lot of ground here
class EDU(Unit):
    """STAC EDU

    A STAC EDU merges information from the unit and discourse
    annotation stages so that you can ignore the distinction
    between the two annotation stages.

    It also tries to be usable as a drop-in substitute for both
    annotations and contexts
    """
    def __init__(self, doc, discourse_anno, unit_anno):
        """
        Parameters
        ----------
        doc : ?
            ?
        discourse_anno : ?
            Annotation from the discourse layer.
        unit_anno : ?
            Annotation from the units layer.
        """
        self._doc = doc
        self._anno = discourse_anno
        self._unit_anno = unit_anno
        unit_anno = unit_anno or discourse_anno
        unit_type = (unit_anno.type if is_edu(unit_anno)
                     else discourse_anno.type)
        super(EDU, self).__init__(discourse_anno.local_id(),
                                  discourse_anno.text_span(),
                                  unit_type,
                                  discourse_anno.features,
                                  discourse_anno.metadata,
                                  discourse_anno.origin)
        # to be fleshed out
        self.turn = None
        self.tstar = None
        self.turn_edus = None
        self.dialogue = None
        self.dialogue_turns = None
        self.doc_turns = None
        self.tokens = None

    def fleshout(self, context):
        """
        second phase of EDU initialisation; fill out contextual info
        """
        self.turn = context.turn
        self.tstar = context.tstar
        self.turn_edus = context.turn_edus
        self.dialogue = context.dialogue
        self.dialogue_turns = context.dialogue_turns
        self.doc_turns = context.doc_turns
        self.tokens = context.tokens

    def speaker(self):
        """
        the speaker associated with the turn surrounding an edu
        """
        return speaker(self.turn)

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
        """What abstract subgrouping the EDU is in (here: turn stars)

        See also
        --------
        educe.stac.context.merge_turn_stars

        Return
        ------
        subgrouping: string
        """
        return self._doc.global_id('t' + str(turn_id(self.tstar)))
# pylint: enable=too-many-instance-attributes


# pylint: disable=no-self-use
class _FakeRootEDU(object):
    """Virtual EDU to represent the notion of a fake root node
    sometimes used in dependency parsing applications
    """
    type = ROOT

    def __init__(self):
        self.turn = self
        self.span = Span(0, 0)

    def text_span(self):
        "Trivial text span"
        return self.span

    def is_left_padding(self):
        "If this is a virtual EDU used in machine learning tasks"
        return True

    def identifier(self):
        """Some kind of identifier string that uniquely identfies the EDU in
        the corpus. Because these are higher level annotations than in the
        Glozz layer we will use the 'local' identifier, which should be the
        same across stages"""
        return ROOT

    def speaker(self):
        "For feature extraction, should not ever really be rendered"
        return None
# pylint: enable=no-self-use

# pylint: disable=invalid-name
FakeRootEDU = _FakeRootEDU()
# pylint: enable=invalid-name


def fuse_edus(discourse_doc, unit_doc, postags):
    """Return a copy of the discourse level doc, merging info from both
    the discourse and units stage.

    All EDUs will be converted to higher level EDUs.

    Notes
    -----
    * The discourse stage is primary in that we work by going over what
      EDUs we find in the discourse stage and trying to enhance them
      with information we find on their units-level equivalents.
      Sometimes (rarely but it happens) annotations can go out of synch.
      EDUs missing on the units stage will be silently ignored (we try
      to make do without them).
      EDUs that were introduced on the units stage but not percolated to
      discourse will also be ignored.

    * We rely on annotation ids to match EDUs from both stages; it's up
      to you to ensure that the annotations are really in synch.

    * This does not constitute a full merge of the documents. For a full
      merge, you would have to bring over other annotations such as
      Resources, `Preference`, `Anaphor`, `Several_resources`, taking
      care all the while to ensure there are no timestamp clashes with
      pre-existing annotations (it's unlikely but best be on the safe
      side if you ever find yourself with automatically generated
      annotations, where all bets are off time-stamp wise).

    Parameters
    ----------
    discourse_doc : GlozzDocument
        Document from the "discourse" stage.
    unit_doc : GlozzDocument
        Document from the "units" stage.
    postags : list of Token
        Sequence of educe tokens predicted by the POS tagger for this
        document.

    Returns
    -------
    doc : GlozzDocument
        Deep copy of the discourse_doc with info from the units stage
        merged in.
    """
    doc = copy.deepcopy(discourse_doc)

    # first pass: create the EDU objects
    annos = sorted([x for x in doc.units if is_edu(x)],
                   key=lambda x: x.span)
    replacements = {}
    for anno in annos:
        unit_anno = None if unit_doc is None else twin_from(unit_doc, anno)
        edu = EDU(doc, anno, unit_anno)
        replacements[anno] = edu

    # second pass: rewrite doc so that annotations that correspond
    # to EDUs are replacement by their higher-level equivalents
    edus = []
    for anno in annos:
        edu = replacements[anno]
        edus.append(edu)
        doc.units.remove(anno)
        doc.units.append(edu)
        for rel in doc.relations:
            if rel.source == anno:
                rel.source = edu
            if rel.target == anno:
                rel.target = edu
        for schema in doc.schemas:
            if anno in schema.units:
                schema.units.remove(anno)
                schema.units.append(edu)

    # fourth pass: flesh out the EDUs with contextual info
    # now the EDUs should work as contexts too
    contexts = Context.for_edus(doc, postags=postags)
    for edu in edus:
        edu.fleshout(contexts[edu])
    return doc
