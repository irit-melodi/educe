"""
The dialogue and turn surrounding an EDU along with some convenient
information about it
"""

import copy
import itertools as itr
import warnings

from educe.annotation import Span
from .annotation import (is_edu, is_cdu, is_dialogue, is_turn,
                         split_turn_text,
                         TURN_TYPES)
from .annotation import speaker as anno_speaker
from .graph import WrappedToken, EnclosureGraph


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


def _blank_out(text, rejects):
    """Return a copy of a text with the indicated regions replaced
    by spaces.

    The resulting text has the same length as the initial one.

    Parameters
    ----------
    text: string

    rejects: [(int,int)]
        A list of (start, end) spans indicating indices for regions
        that should be replaced with spaces
    """
    if not rejects:
        return text
    before = 0
    text2 = ""
    for left, right in rejects:
        text2 += text[before:left]
        text2 += ' ' * (right - left)
        before = right
    text2 += text[before:]
    assert len(text2) == len(text)
    return text2


# FIXME produce proper turn-stars, with type 'Tstar'
# MM: not sure what side-effects this could produce as of 2015-10-20
def merge_turn_stars(doc):
    """Return a copy of the document in which consecutive turns
    by the same speaker have been merged.

    Merging is done by taking the first turn in grouping of
    consecutive speaker turns, and stretching its span over all
    the subsequent turns.

    Additionally turn prefix text (containing turn numbers and
    speakers) from the removed turns are stripped out.
    """
    def prefix_span(turn):
        "given a turn annotation, return the span of its prefix"
        prefix, _ = split_turn_text(doc.text(turn.text_span()))
        start = turn.text_span().char_start
        return start, start + len(prefix)

    doc = copy.deepcopy(doc)
    dialogues = sorted([x for x in doc.units if is_dialogue(x)],
                       key=lambda x: x.text_span())
    rejects = []  # spans for the "deleted" turns' prefixes
    for dia in dialogues:
        dia_turns = sorted(turns_in_span(doc, dia.text_span()),
                           key=lambda x: x.text_span())
        for _, turns in itr.groupby(dia_turns, anno_speaker):
            turns = list(turns)
            tstar = turns[0]
            tstar.span = Span.merge_all(x.text_span() for x in turns)
            rejects.extend(turns[1:])
            for anno in turns[1:]:
                doc.units.remove(anno)
    # pylint: disable=protected-access
    doc._text = _blank_out(doc._text, [prefix_span(x) for x in rejects])
    # pylint: enable=protected-access
    return doc

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

    Parameters
    ----------
    turn:
        the turn surrounding this EDU
    tstar:
        the tstar turn surrounding this EDU (a tstar turn
        is a sort of virtual turn made by merging consecutive
        turns in a dialogue that have the same speaker)
    turn_edus:
        the EDUs in the this turn
    dialogue:
        the dialogue surrounding this EDU
    dialogue_turns:
        all the turns in the dialogue surrounding this EDU
        (non-empty, sorted by first-widest span)
    doc_turns:
        all the turns in the document
    tokens:
        (may not be present): tokens contained within this EDU
    """
    # pylint: disable=too-many-arguments
    def __init__(self,
                 turn,
                 tstar,
                 turn_edus,
                 dialogue,
                 dialogue_turns,
                 doc_turns,
                 tokens=None):
        self.turn = turn
        self.tstar = tstar
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
    def _the(cls, edu, surrounders, types):
        """
        Return the surrounding annotation of the given type.
        We are expecting there to be exactly one such surrounder.
        If none, we consider it worth an exception. If more
        than one, we grit our teeth and move.
        """
        matches = [x for x in surrounders if x.type in types]
        if len(matches) == 1:
            return matches[0]
        else:
            oops = "Was expecting exactly one %s for edu %s" %\
                (types, edu.identifier()) +\
                ", but got %d\nSurrounders found: %s" %\
                (len(matches), [x.identifier() for x in surrounders])
            if matches:
                warnings.warn(oops)
                return matches[0]
            else:
                raise Exception(oops)

    @classmethod
    def _for_edu(cls, enclosure, doc_turns, doc_tstars, edu):
        """Extract the context for a single EDU, but with the benefit of an
        enclosure graph to avoid repeatedly combing over objects

        Parameters
        ----------
        enclosure: EnclosureGraph

        doc_turns: [Unit]
            All turn-level annotations within a document. This is somewhat
            redundant with the enclosure graph, but perhaps more convenient

        doc_tstars: [Unit]
            All turn star annotations within a document. Turn stars are not
            native to the document and have to be computed separately.
            For example, the will not be part of the enclosure graph
            unless you apply a merge_turn_stars on it.

        edu: Unit
        """
        turn = cls._the(edu, enclosure.outside(edu),
                        TURN_TYPES)
        tstar = cls._the(edu, containing(edu.text_span(), doc_tstars),
                         TURN_TYPES)
        t_edus = [x for x in enclosure.inside(turn) if is_edu(x)]
        assert t_edus
        dialogue = cls._the(edu, enclosure.outside(turn),
                            ['Dialogue'])
        d_turns = [x for x in enclosure.inside(dialogue) if is_turn(x)]
        assert d_turns
        tokens = [wrapped.token for wrapped in enclosure.inside(edu)
                  if isinstance(wrapped, WrappedToken)]
        return cls(turn=turn,
                   tstar=tstar,
                   turn_edus=sorted_first_widest(t_edus),
                   dialogue=dialogue,
                   dialogue_turns=sorted_first_widest(d_turns),
                   doc_turns=sorted_first_widest(doc_turns),
                   tokens=tokens)

    @classmethod
    def for_edus(cls, doc, postags=None):
        """
        Return a dictionary of context objects for each EDU in the document

        Returns
        -------
        contexts: dict(educe.glozz.Unit, Context)

            A dictionary with a context For each EDU in the document
        """
        if postags:
            egraph = EnclosureGraph(doc, postags)
        else:
            egraph = EnclosureGraph(doc)
        doc_turns = [x for x in doc.units if is_turn(x)]
        # pylint: disable=bare-except
        # TODO: it would be nice if merge_turn_stars could return a
        # smaller exception for its difficulties
        try:
            tstar_doc = merge_turn_stars(doc)
        except:
            # this comes up with artificial documents generated for test cases
            # but it could also be an issue with incoherent documents
            oops = "Could not merge turn stars for doc: %s" % doc.origin
            warnings.warn(oops)
            tstar_doc = doc
        # pylint: enable=bare-except
        tstars = [x for x in tstar_doc.units if is_turn(x)]
        contexts = {}
        for edu in doc.units:
            if not is_edu(edu):
                continue
            contexts[edu] = cls._for_edu(egraph, doc_turns, tstars, edu)
        return contexts


def speakers(contexts, anno):
    """
    Return a list of speakers of an EDU or CDU (in the textual
    order of the EDUs).
    """
    if is_edu(anno):
        return [contexts[anno].speaker()]
    elif is_cdu(anno):
        edus = sorted([x for x in anno.terminals() if is_edu(x)],
                      key=lambda x: x.text_span())
        return [contexts[x].speaker() for x in edus]
    else:
        raise ValueError('Expected either an EDU or CDU but got: %s | %s' %
                         (type(anno), anno))


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
