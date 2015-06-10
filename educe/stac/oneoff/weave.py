# Author: Eric Kow
# License: BSD3

"""
Combining annotations from an augmented 'source' document (with likely extra
text) with those in a 'target' document. This involves copying missing
annotations over and shifting the text spans of any matching documents
"""

from collections import namedtuple

from educe.annotation import Span
from educe.stac.context import enclosed


class WeaveException(Exception):

    """
    Unexpected alignment issues between the source and target
    document
    """

    def __init__(self, *args, **kw):
        super(WeaveException, self).__init__(*args, **kw)


class Updates(namedtuple('Updates',
                         ['shift_if_ge',
                          'abnormal_src_only',
                          'abnormal_tgt_only',
                          'expected_src_only'])):

    """Expected updates to the target document.

    We expect to see four types of annotation:

    1. target annotations for which there exists a
       source annotation in the equivalent span

    2. target annotations for which there is no equivalent
       source annotation (eg. Resources, Preferences, but also
       annotation moves)

    3. source annotations for which there is at least one target
       annotation at the equivalent span (the mirror to case 1;
       note that these are not represented in this structure
       because we don't need to say much about them)

    4. source annotations for which there is no match in the target side

    5. source annotations that lie in between the matching bits of
       text

    Parameters
    ----------
    shift_if_ge : dict(int, int)
        (case 1 and 2) shift points and offsets for characters
        in the target document (see `shift_spans`)
    abnormal_tgt_only : [Annotation]
        (case 2) annotations that only occur
        in the target document (weird, found in matches)
    abnormal_src_only: [Annotation]
        (case 4) annotations that only occur in the
        source document (weird, found in matches)
    abnormal_src_only [Annotation]
        (case 5) annotations that only occur in the
        source doc (ok, found in gaps)
    """
    def __new__(cls):
        return super(Updates, cls).__new__(cls, {}, [], [], [])

    def map(self, fun):
        """
        Return an `Updates` in which a function has been applied to
        all annotations in this one (eg. useful for previewing),
        and to all spans
        """
        supercls = super(Updates, Updates)
        return supercls.__new__(Updates,
                                self.shift_if_ge,
                                [fun(x) for x in self.abnormal_src_only],
                                [fun(x) for x in self.abnormal_tgt_only],
                                [fun(x) for x in self.expected_src_only])

# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------


def src_gaps(matches):
    """
    Given matches between the source and target document, return the spaces
    between these matches as source offset and size (a bit like the matches).
    Note that we assume that the target document text is a subsequence of the
    source document.
    """
    gaps = []
    last_idx = 0
    for src, _, size in matches:
        if src != last_idx:
            gaps.append((last_idx, src - last_idx))
        last_idx = src + size
    return gaps


def tgt_gaps(matches):
    """
    Given matches between the source and target document, return the spaces
    between these matches as target offset and size (a bit like the matches).
    By rights this should be empty, but you never know
    """
    gaps = []
    last_idx = 0
    for _, tgt, size in matches:
        if tgt != last_idx:
            gaps.append((last_idx, tgt - last_idx))
        last_idx = tgt + size
    return gaps


def check_matches(tgt_doc, matches):
    """
    Check that the target document text is indeed a subsequence of
    the source document text (the source document is expected to be
    "augmented" version of the target with new text interspersed
    throughout)
    """
    tgt_text = tgt_doc.text()

    if not tgt_text:
        return
    elif not matches:
        raise WeaveException('no matches in non-empty target doc')
    elif matches[0].b != 0:
        oops = ('matches ({}) do not start at beginning of target '
                'document <{}>').format(matches[0], tgt_doc.origin)
        raise WeaveException(oops)

    gaps = tgt_gaps(matches)
    if gaps:
        oops = 'there are match gaps in the target document: {}'
        raise WeaveException(oops.format(gaps))

    _, tgt, size = matches[-1]
    if tgt + size != len(tgt_text):
        raise WeaveException('matches do not cover the full target '
                             'document')


def compute_updates(src_doc, tgt_doc, matches):
    """Return updates that would need to be made on the target
    document.

    Given matches between the source and target document, return span
    updates along with any source annotations that do not have an
    equivalent in the target document (the latter may indicate that
    resegmentation has taken place, or that there is some kind of problem)

    Parameters
    ----------
    src_doc : Document
    tgt_doc : Document
    matches : [Match]

    Returns
    -------
    updates: Updates
    """
    res = Updates()

    # case 2 and 5 (to be pruned below)
    res.expected_src_only.extend(src_doc.units)
    res.abnormal_tgt_only.extend(tgt_doc.units)

    # case 1, 2 and 4
    for src, tgt, size in matches:
        tgt_to_src = src - tgt
        res.shift_if_ge[tgt] = tgt_to_src  # case 1 and 2
        src_annos = enclosed(Span(src, src + size), src_doc.units)
        tgt_annos = enclosed(Span(tgt, tgt + size), tgt_doc.units)
        for src_anno in src_annos:
            res.expected_src_only.remove(src_anno)  # prune from case 5
            src_span = src_anno.text_span()
            tgt_equiv = [x for x in tgt_annos
                         if x.text_span().shift(tgt_to_src) == src_span]
            if not tgt_equiv:  # case 4
                res.abnormal_src_only.append(src_anno)
            for tgt_anno in tgt_equiv:  # prun from case 2
                if tgt_anno in res.abnormal_tgt_only:
                    res.abnormal_tgt_only.remove(tgt_anno)

    return res


def shift_char(position, updates):
    """
    Given a character position an updates tuple, return a shifted over
    position which reflects the update.


    The basic idea that we have a set of "shift points" and their
    corresponding offsets. If a character position 'c' occurs after
    one of the points, we take the offset of the largest such point
    and add it to the character.

    Our assumption here is that the update always consists in adding more
    text so offsets are always positive.

    Parameters
    ----------
    position: int
        initial position
    updates: Updates

    Returns
    -------
    int
        shifted position
    """
    points = [x for x in updates.shift_if_ge if position >= x]
    offset = updates.shift_if_ge[max(points)] if points else 0
    assert offset >= 0
    return position + offset


def shift_span(span, updates):
    """
    Given a span and an updates tuple, return a Span
    that is shifted over to reflect the updates

    Parameters
    ----------
    span: Span
    updates: Updates

    Returns
    -------
    span: Span

    See also
    --------
    shift_char: for details on how this works
    """
    start = shift_char(span.char_start, updates)
    # this is to avoid spurious overstretching of the right
    # boundary of an annotation that buts up against the
    # left of a new annotation
    end = 1 + shift_char(span.char_end - 1, updates)
    return Span(start, end)
