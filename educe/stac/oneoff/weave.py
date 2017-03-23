# Author: Eric Kow
# License: BSD3

"""
Combining annotations from an augmented 'source' document (with likely extra
text) with those in a 'target' document. This involves copying missing
annotations over and shifting the text spans of any matching documents
"""

from __future__ import print_function

from collections import namedtuple
from functools import reduce
import itertools
import sys

import numpy as np

from educe.annotation import Span
from educe.stac.annotation import (game_turns, is_dialogue, is_edu,
                                   is_paragraph, is_turn, turn_id,
                                   DIALOGUE_ACTS, RENAMES)
from educe.stac.context import enclosed
from educe.stac.edit.cmd.merge_dialogue import _concatenate_features


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
    abnormal_src_only: [Annotation]
        (case 4) annotations that only occur in the
        source document (weird, found in matches)
    abnormal_tgt_only : [Annotation]
        (case 2) annotations that only occur
        in the target document (weird, found in matches)
    expected_src_only [Annotation]
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


def check_matches(tgt_doc, matches, strict=True):
    """
    Check that the target document text is indeed a subsequence of
    the source document text (the source document is expected to be
    "augmented" version of the target with new text interspersed
    throughout)

    Parameters
    ----------
    tgt_doc :
    matches : list of (int, int, int)
        List of triples (i, j, n) representing matching subsequences:
        a[i:i+n] == b[j:j+n].
        See `difflib.SequenceMatcher.get_matching_blocks`.
    strict : boolean
        If True, raise an exception if there are match gaps in the
        target document, otherwise just print the gaps to stderr.
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
        # we might want to give some slack because gaps can result from
        # manual rewrites that happened here and there in the soclogs
        # e.g. a pair of logical not (&not;) around _ => ^_^
        # in these cases, just print them on stderr for quick checks
        for gap in gaps:
            gap_txt = tgt_text[gap[0]:gap[0] + gap[1]]
            print(u"Match gap in tgt doc ({})\t{}\t{}".format(
                tgt_doc.origin, gap, gap_txt), file=sys.stderr)
        print('Matches: ', matches)
        tgt_turns = set(turn_id(x) for x in tgt_doc.units
                        if turn_id(x) is not None)
        print('Turns: ', sorted(tgt_turns))
        if strict:
            oops = 'there are match gaps in the target document {}: {}'
            raise WeaveException(oops.format(tgt_doc.origin, gaps))

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

    # WIP separate matching procedures for EDUs, turns, paragraphs,
    # dialogues and the rest (seemingly only resources).
    def is_various(annotation):
        """None of {edu, turn, paragraph, dialogue}.

        It seems to capture only Resources (to be confirmed).
        """
        return not(is_edu(annotation) or
                   is_turn(annotation) or
                   is_paragraph(annotation) or
                   is_dialogue(annotation))

    # case 1, 2 and 4
    for src, tgt, size in matches:
        tgt_to_src = src - tgt
        res.shift_if_ge[tgt] = tgt_to_src  # case 1 and 2
        src_annos = enclosed(Span(src, src + size), src_doc.units)
        tgt_annos = enclosed(Span(tgt, tgt + size), tgt_doc.units)
        # WIP separate matching procedures for the different types of
        # annotations
        for anno_type in [is_edu, is_turn, is_paragraph, is_dialogue,
                          is_various]:
            cands_src = [x for x in src_annos if anno_type(x)]
            cands_tgt = [x for x in tgt_annos if anno_type(x)]
            # compute (shifted) spans once
            spans_src = [x.text_span() for x in cands_src]
            spans_tgt = [x.text_span().shift(tgt_to_src) for x in cands_tgt]
            # loop over source annotations
            for src_span, src_anno in zip(spans_src, cands_src):
                res.expected_src_only.remove(src_anno)  # prune from case 5
                tgt_equiv = [tgt_anno for tgt_span, tgt_anno
                             in zip(spans_tgt, cands_tgt)
                             if tgt_span == src_span]
                if not tgt_equiv:  # case 4
                    res.abnormal_src_only.append(src_anno)
                for tgt_anno in tgt_equiv:  # prune from case 2
                    if tgt_anno in res.abnormal_tgt_only:
                        res.abnormal_tgt_only.remove(tgt_anno)

    return res


def update_updates(updates, annos_src, annos_tgt, verbose=0):
    """Update the sets of updates given a match (annos_src, annos_tgt).

    Parameters
    ----------
    updates : Updates
        Summary of extra updates between source and target.
    annos_src : list of Annotation
        Matched annotations from source doc.
    annos_tgt : list of Annotation
        Matched annotations from target doc.
    verbose : int
        Verbosity.

    Returns
    -------
    updates : Updates
        `updates` updated to take the given match into account.
    """
    # the matched target structure shall be kept
    for anno_tgt in annos_tgt:
        if anno_tgt in updates.abnormal_tgt_only:
            updates.abnormal_tgt_only.remove(anno_tgt)
            if verbose:
                print('Remove {} from abnormal_tgt_only'.format(anno_tgt),
                      file=sys.stderr)  # DEBUG
    # the matched source structures are redundant, thus
    # they are neither problematic nor useful
    for anno_src in annos_src:
        # these source structures are neither abnormal
        if anno_src in updates.abnormal_src_only:
            updates.abnormal_src_only.remove(anno_src)
            if verbose:
                print('Remove {} from abnormal_src_only'.format(anno_src),
                      file=sys.stderr)  # DEBUG
        # nor expected to appear only in source
        if anno_src in updates.expected_src_only:
            updates.expected_src_only.remove(anno_src)
            if verbose:
                print('Remove {} from expected_src_only'.format(anno_src),
                      file=sys.stderr)  # DEBUG
    return updates


def stretch_match(updates, src_doc, tgt_doc, doc_span_src, doc_span_tgt,
                  annos_src, annos_tgt, verbose=0):
    """Compute stretch matches between `annos_src` and `annos_tgt`.

    Parameters
    ----------
    updates : Update
    src_doc : Document
    tgt_doc : Document
    doc_span_src : Span
    doc_span_tgt : Span
    annos_src : list of educe.annotation
        Unmatched annotations in `span_src`.
    annos_tgt : list of educe.annotation
        Unmatched annotations in `span_tgt`.
    verbose : int
        Verbosity level

    Returns
    -------
    res : Update
        Possibly trimmed version of `updates`.
    """
    # unmatched structs in src
    cands_src = enclosed(Span(doc_span_src[0], doc_span_src[1]),
                         annos_src)
    spans_src = [anno.text_span() for anno in cands_src]
    # unmatched structs in tgt
    cands_tgt = enclosed(Span(doc_span_tgt[0], doc_span_tgt[1]),
                         annos_tgt)
    spans_tgt = [anno.text_span() for anno in cands_tgt]

    # {one,many} to one match between source and target
    # FIXME separate matching procedures for:
    # * Turn
    # * paragraph
    # * segment (+ segments with different names in units/ ?)
    for span_tgt, cand_tgt in zip(spans_tgt, cands_tgt):
        # span_tgt cast onto src_doc
        shifted_span_tgt = shift_span(span_tgt, updates)

        # 1-1 match on the exact (translated) same span
        src_equiv = [cand_src for span_src, cand_src
                     in zip(spans_src, cands_src)
                     if span_src == shifted_span_tgt]

        # 1-1 stretch match, based on comparing the text of the turns
        # that are common to source and target
        txt_tgt = tgt_doc.text(span=span_tgt).strip()
        # compute the maximal extension of span_src: include whitespaces
        # immediately before
        lctx_src = [src_doc.text(span=Span(span_src.char_start - 10,
                                           span_src.char_start))
                    for span_src in spans_src]
        lpad_src = [len(x) - len(x.rstrip()) for x in lctx_src]
        # ... and after
        rctx_src = [src_doc.text(span=Span(span_src.char_end,
                                           span_src.char_end + 10))
                    for span_src in spans_src]
        rpad_src = [len(x) - len(x.lstrip()) for x in rctx_src]
        # create the corresponding extended spans
        ext_spans_src = [Span(span_src.char_start - lpad,
                              span_src.char_end + rpad)
                         for span_src, lpad, rpad
                         in zip(spans_src, lpad_src, rpad_src)]
        src_equiv_stretch = [cand_src for span_src, ext_span_src, cand_src
                             in zip(spans_src, ext_spans_src, cands_src)
                             if ((txt_tgt ==
                                  hollow_out_missing_turn_text(
                                      src_doc, tgt_doc,
                                      doc_span_src=span_src,
                                      doc_span_tgt=span_tgt
                                  ).replace('\t ', '').replace('\t', '')
                                  .strip()) and
                                 ext_span_src.encloses(shifted_span_tgt))]
        # extend list of 1-1 exact matches with 1-1 stretch matches
        if src_equiv_stretch:
            src_equiv.extend(src_equiv_stretch)
            if verbose:
                print('1-to-1 stretch match: ',
                      [str(x) for x in src_equiv_stretch])
                print('for target annotation: ', cand_tgt)

        if src_equiv:
            updates = update_updates(updates, src_equiv, [cand_tgt],
                                     verbose=verbose)
        else:
            # many to 1 match between source and target
            #
            # search for a sequence of contiguous annotations in source
            # that covers the same span as a single annotation of the
            # same type in target ; this is supposed to capture the
            # result of `stac-edit merge-{dialogue,edu}`
            src_equiv_cands = enclosed(shifted_span_tgt, cands_src)
            src_equiv_seq = sorted(src_equiv_cands, key=lambda x: x.span)
            # if the sequence covers the targeted span
            if ((src_equiv_seq and
                 (src_equiv_seq[0].span.char_start ==
                  shifted_span_tgt.char_start) and
                 (src_equiv_seq[-1].span.char_end ==
                  shifted_span_tgt.char_end))):
                # and has no gap or just whitespaces
                gap_str = ''.join(
                    src_doc.text(span=Span(elt_cur.span.char_end,
                                           elt_nex.span.char_start))
                    for elt_cur, elt_nex
                    in zip(src_equiv_seq[:-1], src_equiv_seq[1:])
                )
                gap_str = gap_str.strip()
                if not gap_str:
                    updates = update_updates(
                        updates, src_equiv_seq, [cand_tgt],
                        verbose=verbose)
                    if verbose:
                        print('Guess: {} results from a merge on {}'.format(
                            str(cand_tgt), [str(x) for x in src_equiv_seq]),
                              file=sys.stderr)

    shifted_spans_tgt = [shift_span(span_tgt, updates)
                         for span_tgt in spans_tgt]  # WIP
    # one to many match between source and target
    for span_src, cand_src in zip(spans_src, cands_src):
        # search for a sequence of contiguous annotations in target
        # that covers the same span as a single annotation of the
        # same type in source ; this is supposed to capture the
        # result of `stac-edit split-{dialogue,edu}`
        tgt_equiv_cands = [(shifted_span_tgt, cand_tgt)
                           for shifted_span_tgt, cand_tgt
                           in zip(shifted_spans_tgt, cands_tgt)
                           if span_src.encloses(shifted_span_tgt)]

        tgt_equiv_seq = sorted(tgt_equiv_cands)
        # if the sequence covers the source span
        if ((tgt_equiv_seq and
             tgt_equiv_seq[0][0].char_start == span_src.char_start and
             tgt_equiv_seq[-1][0].char_end == span_src.char_end)):
            # and has no gap or just whitespaces
            gap_str = ''.join(
                tgt_doc.text(span=Span(elt_cur[1].span.char_end,
                                       elt_nex[1].span.char_start))
                for elt_cur, elt_nex
                in zip(tgt_equiv_seq[:-1], tgt_equiv_seq[1:])
            )
            gap_str = gap_str.strip()
            if not gap_str:
                updates = update_updates(
                    updates, [cand_src], [x[1] for x in tgt_equiv_seq],
                    verbose=verbose
                )
                if verbose:
                    print('Guess: {} results from a split on {}'.format(
                        [str(x[1]) for x in tgt_equiv_seq], str(cand_src)),
                          file=sys.stderr)

    return updates


def find_continuous_seqs(doc, spans, annos):
    """Find continuous sequences of annotations, ignoring whitespaces.

    Parameters
    ----------
    doc : Document
        Annotated document
    spans : list of Span
        Spans that support the annotations
    annos : list of Annotation
        Annotations of interest
    ignore_whitespaces : boolean, optional
        If True, whitespaces are ignored when assessing continuity.

    Returns
    -------
    seqs : list of list of integers
        List of sequences of indices (in annos and spans)
    """
    seqs = [[]]
    seqs[-1].append(0)

    for i, (span_cur, anno_cur, span_nex, anno_nex) in enumerate(zip(
            spans[:-1], annos[:-1], spans[1:], annos[1:])):
        # check for gap
        if span_cur.char_end < span_nex.char_start:
            gap_span = Span(span_cur.char_end, span_nex.char_start)
            # ignore gap if all whitespace
            if doc.text(span=gap_span).strip():
                seqs.append([])
        elif span_cur.char_end > span_nex.char_start:
            raise ValueError('Overlapping annotations?!\n{}\n{}'.format(
                anno_cur, anno_nex))
        # update current (possibly new) sequence
        seqs[-1].append(i + 1)

    return seqs


def stretch_match_many(updates, src_doc, tgt_doc, doc_span_src, doc_span_tgt,
                       annos_src, annos_tgt, verbose=0):
    """Compute n-m stretch matches between `annos_src` and `annos_tgt`.

    Parameters
    ----------
    updates : Update
    src_doc : Document
    tgt_doc : Document
    doc_span_src : Span
    doc_span_tgt : Span
    annos_src : list of educe.annotation
        Unmatched annotations in `span_src`.
    annos_tgt : list of educe.annotation
        Unmatched annotations in `span_tgt`.
    verbose : int
        Verbosity level

    Returns
    -------
    res : Update
        Possibly trimmed version of `updates`.
    """
    # unmatched structs in src
    cands_src = enclosed(Span(doc_span_src[0], doc_span_src[1]),
                         annos_src)
    cands_src = sorted(cands_src, key=lambda x: x.span)
    spans_src = [anno.text_span() for anno in cands_src]
    # unmatched structs in tgt
    cands_tgt = enclosed(Span(doc_span_tgt[0], doc_span_tgt[1]),
                         annos_tgt)
    cands_tgt = sorted(cands_tgt, key=lambda x: x.span)
    spans_tgt = [anno.text_span() for anno in cands_tgt]

    if not (spans_src and spans_tgt):
        return updates

    # many to many match between source and target
    seqs_src = find_continuous_seqs(src_doc, spans_src, cands_src)
    seqs_tgt = find_continuous_seqs(tgt_doc, spans_tgt, cands_tgt)

    # TODO if both sequences span the same text (for common turns), use
    # stretched target annotations
    for seq_src, seq_tgt in zip(seqs_src, seqs_tgt):
        seq_spans_src = [spans_src[i] for i in seq_src]
        seq_annos_src = [cands_src[i] for i in seq_src]
        span_seq_src = Span(seq_spans_src[0].char_start,
                            seq_spans_src[-1].char_end)

        seq_spans_tgt = [spans_tgt[i] for i in seq_tgt]
        seq_annos_tgt = [cands_tgt[i] for i in seq_tgt]
        span_seq_tgt = Span(seq_spans_tgt[0].char_start,
                            seq_spans_tgt[-1].char_end)
        # compare (hollowed) text
        txt_src = src_doc.text(span=span_seq_src)
        txt_src = hollow_out_missing_turn_text(
            src_doc, tgt_doc,
            doc_span_src=span_seq_src,
            doc_span_tgt=span_seq_tgt).replace('\t ', '').replace('\t', '')
        txt_tgt = tgt_doc.text(span=span_seq_tgt)

        if txt_tgt.strip() == txt_src.strip():
            if verbose:
                print('Many-to-many stretch match:\n',
                      'source:\n',
                      '\n'.join(str(x) for x in seq_annos_src),
                      '\ntarget:\n',
                      '\n'.join(str(x) for x in seq_annos_tgt))
            updates = update_updates(updates, seq_annos_src, seq_annos_tgt,
                                     verbose=verbose)

    return updates


UNITS = DIALOGUE_ACTS + list(RENAMES.keys())


def compute_structural_updates(src_doc, tgt_doc, matches, updates, verbose=0):
    """Transfer structural annotations from `tgt_doc` to `src_doc`.

    This is the transposition of `compute_updates` to structural
    units (dialogues only, for the moment).
    """
    # match structural units: transpose to the augmented `src_doc`
    # the structures defined on (a subspan of) a sequence of contiguous
    # match spans in b
    stretch_map = {}
    offset_src = matches[0].a
    offset_tgt = matches[0].b
    for m, m1 in zip(matches[:-1], matches[1:]):
        # the next match is not contiguous on tgt
        if m.b + m.size != m1.b:
            merged_span_src = (offset_src, m.a + m.size)
            merged_span_tgt = (offset_tgt, m.b + m.size)
            stretch_map[merged_span_tgt] = merged_span_src
            # update offsets
            offset_src = m.a
            offset_tgt = m.b
    # add final stretch map
    # m already is matches[-2] at the end of the for loop, but I reset
    # it here, so we don't rely on a leaky variable
    # assert m == matches[-2]
    m = matches[-2]
    merged_span_src = (offset_src, m.a + m.size)
    merged_span_tgt = (offset_tgt, m.b + m.size)
    stretch_map[merged_span_src] = merged_span_tgt

    # stretch match (1-1, n-1, 1-n) segments (EDUs)
    print('Stretch match segments', file=sys.stderr)
    # gather all unmatched units from tgt that can be stretched:
    unmatched_src_annos = set(updates.abnormal_src_only +
                              updates.expected_src_only)
    unmatched_tgt_annos = set(updates.abnormal_tgt_only)
    #
    unmatched_src_segs = [x for x in unmatched_src_annos
                          if x.type.lower() == 'segment']
    unmatched_tgt_segs = [x for x in unmatched_tgt_annos
                          if x.type.lower() == 'segment']
    for span_src, span_tgt in stretch_map.items():
        updates = stretch_match(updates, src_doc, tgt_doc,
                                span_src, span_tgt,
                                unmatched_src_segs, unmatched_tgt_segs,
                                verbose=verbose)

    # stretch match dialogue acts: tgt_units <-> src_segs
    print('Stretch match dialogue acts', file=sys.stderr)
    # gather all unmatched units from tgt that can be stretched:
    unmatched_src_annos = set(updates.abnormal_src_only +
                              updates.expected_src_only)
    unmatched_tgt_annos = set(updates.abnormal_tgt_only)
    #
    unmatched_src_segs = [x for x in unmatched_src_annos
                          if x.type.lower() == 'segment']
    unmatched_tgt_units = [x for x in unmatched_tgt_annos
                           if x.type.lower() in set(y.lower() for y in UNITS)]
    for span_src, span_tgt in stretch_map.items():
        updates = stretch_match(updates, src_doc, tgt_doc,
                                span_src, span_tgt,
                                unmatched_src_segs, unmatched_tgt_units,
                                verbose=verbose)

    # FIXME find a cleaner and more concise formulation, share code
    # with the above
    # dialogues and segments (we'll see if they can be treated the same)
    # n-m matchings: segments (EDUs)
    print('n-m stretch match segments', file=sys.stderr)
    unmatched_src_annos = set(updates.abnormal_src_only +
                              updates.expected_src_only)
    unmatched_tgt_annos = set(updates.abnormal_tgt_only)
    #
    unmatched_src_segs = [x for x in unmatched_src_annos
                          if x.type.lower() == 'segment']
    unmatched_tgt_segs = [x for x in unmatched_tgt_annos
                          if x.type.lower() == 'segment']
    for span_src, span_tgt in stretch_map.items():
        updates = stretch_match_many(updates, src_doc, tgt_doc,
                                     span_src, span_tgt,
                                     unmatched_src_segs, unmatched_tgt_segs,
                                     verbose=verbose)

    # n-m matchings: dialogue acts (tgt_units <-> src_segs)
    print('n-m stretch match on dialogue acts', file=sys.stderr)
    unmatched_src_annos = set(updates.abnormal_src_only +
                              updates.expected_src_only)
    unmatched_tgt_annos = set(updates.abnormal_tgt_only)
    #
    unmatched_src_segs = [x for x in unmatched_src_annos
                          if x.type.lower() == 'segment']
    unmatched_tgt_units = [x for x in unmatched_tgt_annos
                           if x.type.lower() in set(y.lower() for y in UNITS)]
    for span_src, span_tgt in stretch_map.items():
        updates = stretch_match_many(updates, src_doc, tgt_doc,
                                     span_src, span_tgt,
                                     unmatched_src_segs, unmatched_tgt_units,
                                     verbose=verbose)
    return updates


def shift_dialogues(doc_src, doc_res, updates, gen):
    """Transpose dialogue split from target to source document.

    Remove all dialogues from updates.

    Parameters
    ----------
    doc_src : Document
        Source (augmented) document.
    doc_res : Document
        Result document, originally a copy of doc_tgt with unshifted
        annotations. This function modifies `doc_res` by shifting the
        boundaries of its dialogues according to `updates`, and
        stretching the first and last dialogues so as to cover the
        same span as dialogues from `doc_src`.
    updates : set of updates
        Updates computed by `compute_updates`.
    gen: int
        Generation of annotations included in `doc_src` and the output.

    Returns
    -------
    updates : Updates
        Trimmed down set of `updates`: no more dialogue.
    """
    if gen < 3:
        dlgs_src = sorted([x for x in doc_src.units
                           if x.type.lower() == 'dialogue'],
                          key=lambda y: y.span)
        dlgs_res = sorted([x for x in doc_res.units
                           if x.type.lower() == 'dialogue'],
                          key=lambda y: y.span)

        # NEW 2016-06-15 adjust dialogue boundaries
        # for each target dialogue, find the smallest enclosing sequence of
        # source dialogues and map to it
        dlgs_src_beg = np.array([x.span.char_start for x in dlgs_src])
        dlgs_tgt_sbeg = np.array([
            shift_char(x.span.char_start + 1, updates) - 1
            for x in dlgs_res])
        # NB: we need to broadcast (- 1) to get the source dialogue whose
        # start immediately precedes the start of the shifted target
        # dialogue
        tgt2src_beg = (np.searchsorted(dlgs_src_beg, dlgs_tgt_sbeg,
                                       side='right')
                       - 1)
        dlgs_tgt_abeg = dlgs_src_beg[tgt2src_beg]
        # map the shifted end of each target dialogue to the first larger end
        # of a source dialogue
        dlgs_src_end = np.array([x.span.char_end for x in dlgs_src])
        dlgs_tgt_send = np.array([shift_char(x.span.char_end - 1, updates) + 1
                                  for x in dlgs_res])
        tgt2src_end = np.searchsorted(dlgs_src_end, dlgs_tgt_send)
        dlgs_tgt_aend = dlgs_src_end[tgt2src_end]
        # overwrite the adjusted beginning and end, when a game turn
        # overlaps with two different tgt dialogues ;
        # each overlap in the matching signals a split, in the linguistic
        # version, that happens in the middle of a game turn
        for i, (end_cur, beg_nxt) in enumerate(
                zip(tgt2src_end[:-1], tgt2src_beg[1:])):
            if beg_nxt <= end_cur:
                # linguistic turns from the same game turn, in different
                # target dialogues => use the shifted cut point from tgt
                dlgs_tgt_aend[i] = dlgs_tgt_send[i]
                dlgs_tgt_abeg[i + 1] = dlgs_tgt_send[i]
        # find source dialogues included in the shifted+expanded target
        # dialogues
        dlgs_src_matched = reduce(np.union1d,
                                  (np.arange(x_beg, x_end + 1)
                                   for x_beg, x_end
                                   in zip(tgt2src_beg, tgt2src_end)))
        dlgs_src_matched = set(dlgs_src_matched)

        for dlg_res, adj_start, adj_end in zip(
                dlgs_res, dlgs_tgt_abeg, dlgs_tgt_aend):
            dlg_res.span.char_start = adj_start
            dlg_res.span.char_end = adj_end
            # alt: dlg_res.span = Span(start, end)
            #
            # optionally, update timestamp, id, span as in
            # `stac.edit.cmd.split_dialogue.{_actually_split,_set}`

        # remove all source and target dialogues from updates
        for dlg_res in dlgs_res:
            if dlg_res in updates.abnormal_tgt_only:
                updates.abnormal_tgt_only.remove(dlg_res)
        for i, dlg_src in enumerate(dlgs_src):
            if dlg_src in updates.abnormal_src_only:
                updates.abnormal_src_only.remove(dlg_src)
            if ((i in dlgs_src_matched
                 and dlg_src in updates.expected_src_only)):
                # remove matched source dialogues, leave the unmatched
                # ones in expected_src_only, so that they are added later
                # to the woven document
                updates.expected_src_only.remove(dlg_src)

    else:
        # situated version: we can rely on game turns

        # 1. get the identifier of the first and last turn of each game turn
        # in _src: these turns and those in between must end up in the same
        # dialogue
        turns_src = sorted((x for x in doc_src.units if is_turn(x)),
                           key=lambda x: x.span)
        turns_src_tid = np.array([x.features['Identifier']
                                  for x in turns_src])
        turns_src_beg = np.array([x.span.char_start for x in turns_src])
        turns_src_end = np.array([x.span.char_end for x in turns_src])
        # * locate game turns (index of first and last turn)
        gturn_idc = game_turns(doc_src, turns_src, gen=3)
        gturn_idc_beg = np.array(gturn_idc)
        gturn_idc_end = np.array(
            [i - 1 for i in gturn_idc[1:]] + [len(turns_src) - 1])
        # ... and finally
        gturn_src_tid_beg = turns_src_tid[gturn_idc_beg]
        gturn_src_tid_end = turns_src_tid[gturn_idc_end]

        # 2. get the identifier of the first and last turn of each dialogue
        # in _res: these turns and those in between must end up in the same
        # dialogue
        turns_res = sorted((x for x in doc_res.units if is_turn(x)),
                           key=lambda x: x.span)
        turns_res_tid = np.array([x.features['Identifier']
                                  for x in turns_res])
        turns_res_beg = np.array([x.span.char_start for x in turns_res])
        turns_res_end = np.array([x.span.char_end for x in turns_res])
        # align dialogue spans with turn spans
        dlgs_res = sorted((x for x in doc_res.units if is_dialogue(x)),
                          key=lambda x: x.span)
        dlgs_res_beg = np.array([x.span.char_start for x in dlgs_res])
        dlgs_res_end = np.array([x.span.char_end for x in dlgs_res])
        dlgs_res_ti_beg = np.searchsorted(turns_res_beg, dlgs_res_beg)
        dlgs_res_ti_end = np.searchsorted(turns_res_end, dlgs_res_end,
                                          side='right') - 1
        # ... and finally
        dlgs_res_tid_beg = turns_res_tid[dlgs_res_ti_beg]
        dlgs_res_tid_end = turns_res_tid[dlgs_res_ti_end]

        # 3. map _res dialogues to _src game turns
        dlgs_res_ti_beg = np.array(
            [list(turns_src_tid).index(x) for x in dlgs_res_tid_beg])
        dlgs_res_ti_end = np.array(
            [list(turns_src_tid).index(x) for x in dlgs_res_tid_end])
        # * align the beginning (resp. end) indices of game turns and _res
        # dialogues
        dlg2gturn_beg = (np.searchsorted(gturn_idc_beg, dlgs_res_ti_beg,
                                         side='right') - 1)
        dlg2gturn_end = np.searchsorted(gturn_idc_end, dlgs_res_ti_end)
        # * turn indices of the adjusted beginning and end of the _res
        # dialogues
        # initialize along the boundaries of game turns
        dlg_res_src_abeg = [gturn_idc_beg[i] for i in dlg2gturn_beg]
        dlg_res_src_aend = [gturn_idc_end[i] for i in dlg2gturn_end]

        # 4. make dialogue boundaries coincide with game turn boundaries,
        # which occasionally implies merging dialogues from _res

        # * compute a partition on dialogues such that any pair of dialogues
        # overlapping a given game turn are in the same class
        dlg2grp = [0]
        for i, (gturn_end_cur, gturn_beg_nxt) in enumerate(zip(
                dlg2gturn_end[:-1], dlg2gturn_beg[1:])):
            if gturn_beg_nxt <= gturn_end_cur:
                # two _res dialogues overlap a single game turn:
                # put in the same class (to merge dialogues)
                dlg2grp.append(dlg2grp[-1])
            else:
                dlg2grp.append(dlg2grp[-1] + 1)

        # keep one dialogue for each class of dialogues
        for k, g in itertools.groupby(enumerate(dlg2grp),
                                      key=lambda x: x[1]):
            dlg_idc_merged = [x[0] for x in g]
            # adjust boundaries of the first dialogue of the group
            # index of first and last dialogues
            di_beg = dlg_idc_merged[0]
            di_end = dlg_idc_merged[-1]
            # index of first and last turns of these dialogues
            ti_beg = dlg_res_src_abeg[di_beg]
            ti_end = dlg_res_src_aend[di_end]
            # keep first dialogue, update its features to include those
            # from the other dialogues in the same class
            new_dlg = dlgs_res[di_beg]
            new_dlg.span.char_start = turns_src_beg[ti_beg]
            new_dlg.span.char_end = turns_src_end[ti_end]
            dlgs_res_merged = [dlgs_res[i] for i in dlg_idc_merged]
            for feat in ['Trades', 'Gets', 'Dice_rolling']:
                new_dlg.features[feat] = _concatenate_features(
                    dlgs_res_merged, feat)
            # remove merged dialogues [1:] from doc_res
            for i in dlg_idc_merged[1:]:
                dlg_res = dlgs_res[i]
                doc_res.units.remove(dlg_res)

        # transfer each unmatched (non-overlapping) game turn as a dialogue
        # (which already exists in doc_src)
        gturns_matched = reduce(np.union1d,
                                (np.arange(x_beg, x_end + 1)
                                 for x_beg, x_end
                                 in zip(dlg2gturn_beg, dlg2gturn_end)))
        gturns_matched = set(gturns_matched)
        # each dialogue in doc_src is a game turn
        dlgs_src = sorted((x for x in doc_src.units if is_dialogue(x)),
                          key=lambda x: x.span)
        # remove all source and target dialogues from updates
        for dlg_res in dlgs_res:
            if dlg_res in updates.abnormal_tgt_only:
                updates.abnormal_tgt_only.remove(dlg_res)
        for i, dlg_src in enumerate(dlgs_src):
            if dlg_src in updates.abnormal_src_only:
                updates.abnormal_src_only.remove(dlg_src)
            if ((i in gturns_matched
                 and dlg_src in updates.expected_src_only)):
                # remove matched source dialogues, leave the unmatched
                # ones in expected_src_only, so that they are added later
                # to the woven document
                updates.expected_src_only.remove(dlg_src)

    return updates


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


def shift_span(span, updates, stretch_right=False):
    """
    Given a span and an updates tuple, return a Span
    that is shifted over to reflect the updates

    Parameters
    ----------
    span: Span
    updates: Updates
    stretch_right : boolean, optional
        If True, stretch the right boundary of an annotation that buts up
        against the left of a new annotation. This is recommended for
        annotations that should fully cover a given span, like dialogues
        for documents.

    Returns
    -------
    span: Span

    See also
    --------
    shift_char: for details on how this works
    """
    start = shift_char(span.char_start, updates)
    if stretch_right:
        end = shift_char(span.char_end, updates)
    else:
        # this is to avoid spurious overstretching of the right
        # boundary of an annotation that buts up against the
        # left of a new annotation
        end = 1 + shift_char(span.char_end - 1, updates)
    return Span(start, end)


def hollow_out_missing_turn_text(src_doc, tgt_doc,
                                 doc_span_src=None, doc_span_tgt=None):
    """Return a version of the source text where all characters in turns
    present in `src_doc` but not in `tgt_doc` are replaced with a
    nonsense char (tab).

    Parameters
    ----------
    src_doc : Document
    tgt_doc : Document
    doc_span_src : Span, optional
    doc_span_tgt : Span, optional

    Notes
    -----
    We use difflib's SequenceMatcher to compare the original (but annotated)
    corpus against the augmented corpus containing nonplayer turns. This
    gives us the ability to shift annotation spans into the appropriate
    place within the augmented corpus. By rights the diff should yield only
    inserts (of the nonplayer turns). But if the inserted text should happen
    to have the same sorts of substrings as you might find in the rest of
    corpus, the diff algorithm can be fooled.
    """
    # docstring followup:
    #
    # That said, since we know exactly what things we expect to have inserted,
    # it's not clear to me why we are using diff and not just computing the
    # shifts off the nonplayer turns. Was I being lazy? Did I just not work
    # out this was possible? Was I trying to be robust? It could also have
    # something to do with managing the extra bits of whitespace around the
    # new nonplayer turns.  To simplify...
    units_tgt = tgt_doc.units
    if doc_span_tgt is not None:
        units_tgt = [x for x in units_tgt
                     if doc_span_tgt.encloses(x.span)]
    units_src = src_doc.units
    if doc_span_src is not None:
        units_src = [x for x in units_src
                     if doc_span_src.encloses(x.span)]

    # we can't use the API one until we update it to account for the
    # fancy new identifiers
    tgt_turns = set(x.features['Identifier']
                    for x in units_tgt
                    if x.features.get('Identifier'))
    np_spans = [x.text_span() for x in units_src
                if (x.features.get('Identifier') and
                    x.features['Identifier'] not in tgt_turns)]

    # merge consecutive nonplayer turns
    merged_np_spans = []
    if np_spans:
        current = None
        for span in sorted(np_spans):
            if not current:
                current = span
                continue
            elif span.char_start == current.char_end + 1:
                current = current.merge(span)
            else:
                merged_np_spans.append(current)
                current = span
        merged_np_spans.append(current)

    if doc_span_src is not None:
        orig = src_doc.text(span=doc_span_src)
    else:
        orig = src_doc.text()
    res = ''
    last = 0
    for span in merged_np_spans:
        res += orig[last:span.char_start]
        res += '\t' * (span.char_end - span.char_start)
        last = span.char_end
    res += orig[last:]
    return res
