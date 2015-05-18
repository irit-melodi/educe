# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Merge adjacent EDUs
"""

from __future__ import print_function
from collections import namedtuple
import copy
import sys

from educe.stac.context import (edus_in_span)
import educe.stac

from educe.stac.util.annotate import show_diff, annotate_doc
from educe.stac.util.glozz import (TimestampCache,
                                   set_anno_author,
                                   set_anno_date)
from educe.stac.util.args import\
    add_usual_input_args,\
    add_usual_output_args,\
    add_commit_args,\
    read_corpus_with_unannotated,\
    get_output_dir, announce_output_dir,\
    comma_span
from educe.stac.util.doc import\
    narrow_to_span, enclosing_span, retarget
from educe.stac.util.output import save_document
from .split_edu import\
    _AUTHOR, _SPLIT_PREFIX

NAME = 'merge-edu'
# pylint: disable=fixme
_MERGE_PREFIX = _SPLIT_PREFIX
# pylint: enable=fixme


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True)
    parser.add_argument('--annotator', metavar='PY_REGEX',
                        required=True,  # should limit annotator
                        help='annotator')
    parser.add_argument('--span', metavar='SPAN', type=comma_span,
                        required=True,
                        help='eg. 347,363')
    add_usual_output_args(parser)
    add_commit_args(parser)
    parser.set_defaults(func=main)


def _actually_merge(tcache, edus, doc):
    """
    Given a timestamp cache, a document and a collection of edus,
    replace the edus with a single merged edu in the document

    Anything that points to one of the EDUs should point
    instead to the new edu.

    Anything which points exclusively to EDUs in the span
    should be deleted (or signaled?)

    Annotations and features should be merged
    """

    def one_or_join(strs):
        "Return element if singleton, otherwise moosh together"
        strs = [x for x in strs if x is not None]
        return list(strs)[0] if len(strs) == 1\
            else _MERGE_PREFIX + "/".join(strs)

    if not edus:
        return
    new_edu = copy.deepcopy(edus[0])
    new_edu.span = enclosing_span([x.text_span() for x in edus])
    stamp = tcache.get(new_edu.span)
    set_anno_date(new_edu, stamp)
    set_anno_author(new_edu, _AUTHOR)

    if doc.origin.stage == 'units':
        new_edu.type = one_or_join(frozenset(x.type for x in edus))
        # feature keys for all edus
        all_keys = frozenset(x for edu in edus for x in edu.features.keys())
        for key in all_keys:
            old_values = frozenset(x.features.get(key) for x in edus)
            new_edu.features[key] = one_or_join(old_values)

    # in-place replacement
    for i, _ in enumerate(doc.units):
        if doc.units[i] in edus:
            doc.units[i] = new_edu
            break

    for edu in edus:
        if edu in doc.units:
            doc.units.remove(edu)
        retarget(doc, edu.local_id(), new_edu)


def _merge_edus(tcache, span, doc):
    """
    Find any EDUs within the given span in the document
    and merge them into a single one.

    The EDUs should stretch from the beginning to the end of
    the span (gaps OK).

    The output EDU should have the same ID in all documents
    """
    edus = edus_in_span(doc, span)
    if not edus:
        sys.exit("No EDUs in span %s" % span)

    espan = enclosing_span([x.text_span() for x in edus])
    if espan != span:
        sys.exit("EDUs in do not cover full span %s [only %s]" %
                 (span, espan))
    _actually_merge(tcache, edus, doc)


def _mini_diff(k, old_doc, new_doc, span):
    """
    Return lines of text to be printed out, showing how the EDU
    split affected the text
    """
    mini_old_doc = narrow_to_span(old_doc, span)
    mini_new_doc = narrow_to_span(new_doc, span)
    return ["======= MERGE EDUS %s ========" % (k),
            "...",
            show_diff(mini_old_doc, mini_new_doc),
            "...",
            ""]


CommitInfo = namedtuple("CommitTuple", "key annotator before after span")


def commit_msg(info):
    """
    Generate a commit message describing the operation
    we just did
    """
    k = info.key
    turns = [x for x in info.before.units if educe.stac.is_turn(x) and
             x.text_span().encloses(info.span)]
    if turns:
        turn = turns[0]
        tspan = turn.text_span()
        ttext = info.before.text(tspan)
        prefix_b = educe.stac.split_turn_text(ttext)[0]
    else:
        tspan = info.span
        prefix_b = "    "
    prefix_a = "==> ".rjust(len(prefix_b))

    def anno(doc, prefix, tspan):
        "pad text segment as needed"

        prefix_t = "..."\
            if tspan.char_start + len(prefix) < info.span.char_start\
            else ""
        suffix_t = "..."\
            if tspan.char_end > info.span.char_end + 1\
            else ""
        return "".join([prefix,
                        prefix_t,
                        annotate_doc(doc, span=info.span),
                        suffix_t])

    lines = ["%s_%s: scary edit (merge EDUs)" % (k.doc, k.subdoc),
             "",
             anno(info.before, prefix_b, tspan),
             anno(info.after, prefix_a, tspan),
             "",
             "NB: only unannotated and %s are modified" % info.annotator]
    return "\n".join(lines)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus_with_unannotated(args)
    tcache = TimestampCache()
    output_dir = get_output_dir(args)
    commit_info = None
    for k in corpus:
        old_doc = corpus[k]
        new_doc = copy.deepcopy(old_doc)
        _merge_edus(tcache, args.span, new_doc)
        diffs = _mini_diff(k, old_doc, new_doc, args.span)
        print("\n".join(diffs).encode('utf-8'), file=sys.stderr)
        save_document(output_dir, k, new_doc)
        # for commit message generation
        commit_info = CommitInfo(key=k,
                                 annotator=args.annotator,
                                 before=old_doc,
                                 after=new_doc,
                                 span=args.span)
    announce_output_dir(output_dir)
    if commit_info and not args.no_commit_msg:
        print("-----8<------")
        print(commit_msg(commit_info))
