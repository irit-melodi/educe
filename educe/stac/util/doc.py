# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Utilities for large-scale changes to educe documents,
for example, moving a chunk of text from one document
to another
"""

from __future__ import print_function
from collections import defaultdict
import copy

from educe.annotation import Unit, Span
from educe.util import concat_l
import educe.stac

from .glozz import (anno_id_from_tuple,
                    anno_id_to_tuple,
                    anno_author,
                    anno_date,
                    set_anno_date)


class StacDocException(Exception):
    """
    An exception that arises from trying to manipulate a stac
    document (typically moving things around, etc)
    """
    def __init__(self, msg):
        super(StacDocException, self).__init__(msg)


def _set_doc_parts(doc, parts):
    """
    Update a document so that it has annotations from all
    the given subdocuments.

    Note that no attention is paid to annotation ids, spans,
    etc. It's up to you to ensure that everything is kosher.
    """
    doc.units = concat_l(x.units for x in parts)
    doc.relations = concat_l(x.relations for x in parts)
    doc.schemas = concat_l(x.schemas for x in parts)


def evil_set_id(anno, author, date):
    """
    This is a bit evil as it's using undocumented functionality
    from the educe.annotation.Standoff object
    """
    anno._anno_id = anno_id_from_tuple((author, date))
    anno.metadata['author'] = author
    anno.metadata['creation-date'] = str(date)


def evil_set_text(doc, text):
    """
    This is a bit evil as it's using undocumented functionality
    from the educe.annotation.Document object
    """
    doc._text = text


def retarget(doc, old_id, new_anno):
    """
    Replace all links to the old (unit-level) annotation
    with links to the new one.

    We refer to the old annotation by id, but the new
    annotation must be passed in as an object. It must
    also be either an EDU or a CDU.

    Return True if we replaced anything
    """
    new_id = new_anno.local_id()
    new_is_cdu = educe.stac.is_cdu(new_anno)

    replaced = False
    for rel in doc.relations:
        if rel.span.t1 == old_id:
            rel.span.t1 = new_id
            rel.source = new_anno
            replaced = True
        if rel.span.t2 == old_id:
            rel.span.t2 = new_id
            rel.target = new_anno
            replaced = True
    for schema in doc.schemas:
        if old_id in schema.units:
            schema.units = set(schema.units)
            schema.units.remove(old_id)
            replaced = True
            if new_is_cdu:
                schema.schemas = schema.schemas | set(new_id)
            else:
                schema.units.add(new_id)
    return replaced


def shift_annotations(doc, offset, point=None):
    """
    Return a deep copy of a document such that all annotations
    have been shifted by an offset.

    If shifting right, we pad the document with whitespace
    to act as filler. If shifting left, we cut the text.

    If a shift point is specified and the offset is positive,
    we only shift annotations that are to the right of the
    point. Likewise if the offset is negative, we only shift
    those that are to the left of the point.
    """
    def is_moveable(anno):
        "If the annotation should be shifted"
        if point is None:
            return True
        elif offset >= 0:
            return anno.text_span().char_start >= point
        else:
            return anno.text_span().char_end <= point

    def shift(anno):
        "Shift a single annotation"
        if offset != 0 and isinstance(anno, Unit) and is_moveable(anno):
            anno.span = anno.span.shift(offset)
        return anno

    if offset > 0:
        padding = " " * offset
        txt2 = padding + doc.text()
    else:
        start = 0 - offset
        txt2 = doc.text()[start:]
    doc2 = copy.deepcopy(doc)
    evil_set_text(doc2, txt2)
    doc2.units = [shift(x) for x in doc2.units]
    doc2.schemas = [shift(x) for x in doc2.schemas]
    doc2.relations = [shift(x) for x in doc2.relations]
    return doc2


def compute_renames(avoid, incoming):
    """
    Given two sets of documents (i.e. corpora), return a dictionary
    which would allow us to rename ids in `incoming` so that
    they do not overlap with those in `avoid`.

    :rtype `author -> date -> date`
    """
    dates = defaultdict(list)
    renames = defaultdict(dict)
    for doc1 in avoid.values():
        for anno in doc1.annotations():
            author = anno_author(anno)
            date = anno_date(anno)
            dates[author].append(date)
    min_dates = {k: min(v) for k, v in dates.items()}
    max_dates = {k: max(v) for k, v in dates.items()}
    for doc2 in incoming.values():
        for anno in doc2.annotations():
            author = anno_author(anno)
            old_date = anno_date(anno)
            if ((author in dates and
                 old_date in dates[author] and
                 not (author in renames and
                      old_date in renames[author]))):
                if old_date < 0:
                    new_date = min_dates[author] - 1
                    min_dates[author] = new_date
                else:
                    new_date = max_dates[author] + 1
                    max_dates[author] = new_date
                dates[author].append(new_date)
                renames[author][old_date] = new_date
    return renames


def narrow_to_span(doc, span):
    """
    Return a deep copy of a document with only the text and
    annotations that are within the span specified by portion.
    """
    def slice_annos(annos):
        "Select annotations within a span"
        return [x for x in annos if span.encloses(x.text_span())]

    offset = 0 - span.char_start
    doc2 = copy.deepcopy(doc)
    doc2.units = slice_annos(doc2.units)
    doc2.schemas = slice_annos(doc2.schemas)
    doc2.relations = slice_annos(doc2.relations)
    # NB: shift_annotations() does a deepcopy too
    doc2 = shift_annotations(doc2, offset)
    evil_set_text(doc2, doc.text()[span.char_start:span.char_end])
    return doc2


def split_doc(doc, middle):
    """
    Given a split point, break a document into two pieces.
    If the split point is None, we take the whole document
    (this is slightly different from having -1 as a split
    point)

    Raise an exception if there are any annotations that span the point.

    Parameters
    ----------
    doc : Document
        The document we want to split.
    middle : int
        Split point.

    Returns
    -------
    doc_prefix : Document
        Deep copy of `doc` restricted to span [:middle]
    doc_suffix : Document
        Deep copy of `doc` restricted to span [middle:] ; the span of each
        annotation is shifted to match the new text.
    """
    doc_len = len(doc.text())
    if middle < 0:
        middle = doc_len + middle

    def straddles(point, span):
        """
        True if the point is somewhere in the middle of the span
        (sitting at right edge doesn't count).

        Note that this is not the same as checking for enclosure
        because we do not include the rightward edge
        """
        if span is None:
            return False
        return span.char_start < point and span.char_end > point

    leftovers = [x for x in doc.annotations()
                 if straddles(middle, x.text_span())]

    if leftovers:
        oops = ("Can't split document [{origin}] at {middle} because it is "
                "straddled by the following annotations:\n"
                "{annotations}\n"
                "Either split at a different place or remove the annotations")
        leftovers = [' * %s %s' % (x.text_span(), x) for x in leftovers]
        raise StacDocException(oops.format(origin=doc.origin,
                                           middle=middle,
                                           annotations='\n'.join(leftovers)))

    prefix = Span(0, middle)
    suffix = Span(middle, doc_len)
    doc_prefix = narrow_to_span(doc, prefix)
    doc_suffix = narrow_to_span(doc, suffix)
    return doc_prefix, doc_suffix


def rename_ids(renames, doc):
    """
    Return a deep copy of a document, with ids reassigned
    according to the renames dictionary
    """
    def adjust(pointer):
        """Given an annotation id string, return its rename
        if applicable, else the string
        """
        author, date = anno_id_to_tuple(pointer)
        if author in renames and date in renames[author]:
            date2 = renames[author][date]
            return anno_id_from_tuple((author, date2))
        else:
            return pointer

    doc2 = copy.deepcopy(doc)
    for anno in doc2.annotations():
        author = anno_author(anno)
        date = anno_date(anno)
        if author in renames and date in renames[author]:
            new_date = renames[author][date]
            set_anno_date(anno, new_date)
            evil_set_id(anno, author, new_date)

    # adjust pointers
    for anno in doc2.relations:
        anno.span.t1 = adjust(anno.span.t1)
        anno.span.t2 = adjust(anno.span.t2)
    for anno in doc2.schemas:
        anno.units = set(adjust(x) for x in anno.units)
        anno.relations = set(adjust(x) for x in anno.relations)
        anno.schemas = set(adjust(x) for x in anno.schemas)
    return doc2


def move_portion(renames, src_doc, tgt_doc,
                 src_split,
                 tgt_split=-1):
    """Move part of the source document into the target document.

    This returns an updated copy of both the source and target
    documents.

    This can capture a couple of patterns:

        * reshuffling the boundary between the target and source
          document (if `tgt | src1 src2 ==> tgt src1 | src2`)
          (`tgt_split = -1`)
        * prepending the source document to the target
          (`src | tgt ==> src tgt`; `src_split=-1; tgt_split=0`)
        * inserting the whole source document into the other
          (`tgt1 tgt2 + src ==> tgt1 src tgt2`; `src_split=-1`)

    There's a bit of potential trickiness here:

        * we'd like to preserve the property that text has a single
          starting and ending space (no real reason just seems safer
          that way)
        * if we're splicing documents together particularly at their
          respective ends, there's a strong off-by-one risk because
          some annotations span the whole text (whitespace and all),
          particularly dialogues

    Parameters
    ----------
    renames : TODO
        TODO
    src_doc : Document
        Source document
    tgt_doc : Document
        Target document
    src_split : int
        Split point for `src_doc`.
    tgt_split : int
        Split point for `tgt_doc`.

    Returns
    -------
    new_src_doc : Document
        TODO
    new_tgt_doc : Document
        TODO
    """
    def snippet(txt, point, width=30):
        "text fragment with highlight"
        if point > 0:
            return u"[{0}]{1}...".format(txt[point],
                                         txt[point+1:point+width])
        else:
            return u"...{0}[{1}]".format(txt[point-width:point],
                                         txt[point])

    tgt_text = tgt_doc.text()
    src_text = src_doc.text()

    if not tgt_text[tgt_split] == ' ':
        oops = u"Target text does not start with a space at the " + \
               u"insertion point:\n{}...".format(snippet(tgt_text, tgt_split))
        raise StacDocException(oops)
    if not src_text[0] == ' ':
        oops = "Source text does not start with a space\n" +\
                snippet(src_text, 0)
        raise StacDocException(oops)
    if not src_text[src_split] == ' ':
        oops = "Source text does not have a space at its split point\n" +\
               snippet(src_text, src_split)
        raise StacDocException(oops)

    snipped, new_src_doc = split_doc(src_doc, src_split)
    prefix_text = tgt_text[:tgt_split]
    middle_text = snipped.text()
    suffix_text = tgt_text[tgt_split:]

    middle = rename_ids(renames,
                        shift_annotations(snipped, len(prefix_text)))

    if tgt_split >= 0:
        new_tgt_doc = shift_annotations(tgt_doc, len(middle_text),
                                        point=tgt_split)
    else:
        new_tgt_doc = copy.deepcopy(tgt_doc)
    _set_doc_parts(new_tgt_doc, [new_tgt_doc, middle])
    evil_set_text(new_tgt_doc, prefix_text + middle_text + suffix_text)

    return new_src_doc, new_tgt_doc


def strip_fixme(act):
    """
    Remove the fixme string from a dialogue act annotation.
    These were automatically inserted when there is an annotation
    to review. We shouldn't see them for any use cases like feature
    extraction though.

    See `educe.stac.dialogue_act` which returns the set of dialogue
    acts for each annotation (by rights should be singleton set,
    but there used to be more than one, something we want to phase
    out?)
    """
    # pylint: disable=fixme
    pref = "FIXME:"
    # pylint: enable=fixme
    return act[len(pref):] if act.startswith(pref) else act


def unannotated_key(key):
    """
    Given a corpus key, return a copy of that equivalent key
    in the unannotated portion of the corpus (the parser
    outputs objects that are based in unannotated)
    """
    ukey = copy.copy(key)
    ukey.stage = 'unannotated'
    ukey.annotator = None
    return ukey
