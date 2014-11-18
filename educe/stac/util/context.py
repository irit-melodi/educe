"""
The dialogue and turn surrounding an EDU along with some convenient
information about it
"""

import warnings
from ..annotation import is_edu, is_turn
from ..annotation import speaker as anno_speaker
from ..graph import WrappedToken, EnclosureGraph


# TODO: refactor with educe.stac.graph
def sorted_first_widest(nodes):
    """
    Given a list of nodes, return the nodes ordered by their starting point,
    and in case of a tie their inverse width (ie. widest first).
    """
    def from_span(span):
        """
        negate the endpoint so that if we have a tie on the starting
        point, the widest span comes first
        """
        if span:
            return (span.char_start, 0 - span.char_end)
        else:
            return None
    return sorted(nodes, key=lambda x: from_span(x.text_span()))

# ---------------------------------------------------------------------
# contexts
# ---------------------------------------------------------------------


# pylint: disable=too-few-public-methods
class Context(object):
    """
    Representation of the surrounding context for an EDU,
    basically the relevant enclosing annotations: turns,
    dialogues. The idea is potentially extend this to a
    somewhat richer notion of context, including things
    like a sentence count, etc.

    * turn     - the turn surrounding this EDU
    * turn_edus - the EDUs in the this turn
    * dialogue - the dialogue surrounding this EDU
    * dialogue_turns - all the turns in the dialogue surrounding this EDU
                       (non-empty, sorted by first-widest span)
    * doc_turns - all the turns in the document
    * tokens   - (may not be present): tokens contained within this EDU

    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 turn, turn_edus,
                 dialogue, dialogue_turns,
                 doc_turns,
                 tokens=None):
        self.turn = turn
        self.turn_edus = turn_edus
        self.dialogue = dialogue
        self.dialogue_turns = dialogue_turns
        self.doc_turns = doc_turns
        self.tokens = tokens
    # pylint: enable=too-many-arguments

    def speaker(self):
        """
        the speaker associated with the turn surrounding an edu
        """
        return anno_speaker(self.turn)

    @classmethod
    def _the(cls, edu, surrounders, typ):
        """
        Return the surrounding annotation of the given type.
        We are expecting there to be exactly one such surrounder.
        If none, we we consider it worth an exception. If more
        than one, we grit our teeth and move.
        """
        matches = [x for x in surrounders if x.type == typ]
        if len(matches) == 1:
            return matches[0]
        else:
            oops = "Was expecting exactly one %s for edu %s" %\
                (typ, edu.identifier()) +\
                ", but got %d\nSurrounders found: %s" %\
                (len(matches), map(str, surrounders))
            if matches:
                warnings.warn(oops)
                return matches[0]
            else:
                raise Exception(oops)

    @classmethod
    def _for_edu(cls, enclosure, doc_turns, edu):
        """
        Extract the context for a single EDU, but with the benefit of an
        enclosure graph to avoid repeatedly combing over objects
        """
        turn = cls._the(edu, enclosure.outside(edu), 'Turn')
        t_edus = list(filter(is_edu, enclosure.inside(turn)))
        assert t_edus
        dialogue = cls._the(edu, enclosure.outside(turn), 'Dialogue')
        d_turns = list(filter(is_turn, enclosure.inside(dialogue)))
        assert d_turns
        tokens = [wrapped.token for wrapped in enclosure.inside(edu)
                  if isinstance(wrapped, WrappedToken)]
        return cls(turn, sorted_first_widest(t_edus),
                   dialogue, sorted_first_widest(d_turns),
                   sorted_first_widest(doc_turns),
                   tokens=tokens)

    @classmethod
    def for_edus(cls, doc, postags=None):
        """
        Return a dictionary of context objects for each EDU in the document

        :rtype dict(educe.glozz.Unit, Context)
        """
        if postags:
            egraph = EnclosureGraph(doc, postags)
        else:
            egraph = EnclosureGraph(doc)
        doc_turns = list(filter(is_turn, doc.units))
        contexts = {}
        for edu in filter(is_edu, doc.units):
            contexts[edu] = cls._for_edu(egraph, doc_turns, edu)
        return contexts

    @classmethod
    def for_corpus(cls, corpus):
        """
        Return a dictionary of context objects for each EDU in all the
        documents of the corpus.

        Does not include the possibility of including postags in the
        context

        :rtype dict(educe.glozz.Unit, Context)
        """
        contexts = {}
        for key in corpus:
            doc = corpus[key]
            egraph = EnclosureGraph(doc)
            doc_turns = list(filter(is_turn, doc.units))
            for edu in filter(is_edu, doc.units):
                contexts[edu] = cls._for_edu(egraph, doc_turns, edu)
        return contexts


def enclosed(span, annos):
    """
    Given an iterable of standoff, pick just those that are
    enclosed by the given span (ie. are smaller and within)
    """
    return [anno for anno in annos if span.encloses(anno.span)]


def containing(span, annos):
    """
    Given an iterable of standoff, pick just those that
    enclose/contain the given span (ie. are bigger and around)
    """
    return [anno for anno in annos if anno.span.encloses(span)]


def edus_in_span(doc, span):
    """
    Given an document and a text span return the EDUs the
    document contains in that span
    """
    return [anno for anno in enclosed(span, doc.units)
            if is_edu(anno)]


def turns_in_span(doc, span):
    """
    Given a document and a text span, return the turns that the
    document contains in that span
    """
    return [anno for anno in enclosed(span, doc.units)
            if is_turn(anno)]
