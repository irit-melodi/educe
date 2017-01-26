# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""Split a dialogue annotation into two

Warning: the second dialogue annotation will have no
trades or dicerolls
"""

from __future__ import print_function
from collections import namedtuple
import copy
import sys

from educe.annotation import Span
import educe.stac as st

from educe.stac.annotation import parse_turn_id
from educe.stac.util.annotate import show_diff, annotate_doc
from educe.stac.util.args import (add_usual_input_args, add_usual_output_args,
                                  add_commit_args,
                                  read_corpus, get_output_dir,
                                  announce_output_dir)
from educe.stac.util.glozz import (TimestampCache,
                                   set_anno_author, set_anno_date)
from educe.stac.util.doc import narrow_to_span
from educe.stac.util.output import save_document

_AUTHOR = 'stacutil'


def _mini_diff(k, args, old_doc, new_doc, span):
    """
    Return lines of text to be printed out, showing how the nudge
    affected the text
    """
    mini_old_doc = narrow_to_span(old_doc, span)
    mini_new_doc = narrow_to_span(new_doc, span)
    return ["======= SPLIT AT TURN {} in {} ========".format(args.turn, k),
            "...",
            show_diff(mini_old_doc, mini_new_doc),
            "...",
            ""]


def _set(tcache, span, anno):
    """Assign an annotation an id/span according to the timestamp cache"""
    stamp = tcache.get(span)
    set_anno_date(anno, stamp)
    set_anno_author(anno, _AUTHOR)
    anno.span = span


def _actually_split(tcache, doc, dialogue, turn):
    """Split the dialogue before the given turn.
    """
    dspan = dialogue.text_span()
    tspan = turn.text_span()
    span1 = Span(dspan.char_start, tspan.char_start - 1)
    span2 = Span(tspan.char_start - 1, dspan.char_end)
    dialogue1 = dialogue
    dialogue2 = copy.deepcopy(dialogue)
    _set(tcache, span1, dialogue1)
    _set(tcache, span2, dialogue2)
    doc.units.append(dialogue2)
    dialogue2.features = {}


def _the(desc, items):
    """Return the first element of a sequence if it's a singleton

    Die if there are none or more than one
    """
    items = list(items)
    if not items:
        sys.exit("Could not find " + desc)
    elif len(items) > 1:
        sys.exit("Found more than one " + desc + " (%d)" % len(items))
    else:
        return items[0]


def _split_dialogue(tcache, doc, tid):
    """Split a dialogue at a turn

    Turns at or after the given tid are pushed into a new empty
    dialogue.

    Returns
    -------
    Span for the dialogue that was split
    """

    wanted_t = 'turn {}'.format(tid)
    wanted_d = 'dialogue for ' + wanted_t
    turn = _the(wanted_t, [x for x in doc.units if st.is_turn(x) and
                           st.turn_id(x) == tid])
    dialogue = _the(wanted_d, [x for x in doc.units if st.is_dialogue(x) and
                               x.encloses(turn)])
    dspan = dialogue.text_span()
    _actually_split(tcache, doc, dialogue, turn)
    return dspan


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

NAME = 'split-dialogue'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True)
    add_usual_output_args(parser, default_overwrite=True)
    add_commit_args(parser)
    parser.add_argument('turn', metavar='TURN', type=parse_turn_id,
                        help='turn number')
    parser.set_defaults(func=main)


CommitInfo = namedtuple("CommitTuple", "key before after span tid")


def commit_msg(info):
    """
    Generate a commit message describing the operation
    we just did
    """
    k = info.key
    mini_new_doc = narrow_to_span(info.after, info.span)

    lines = ["{}_{}: split dialogue before turn {}".format(
        k.doc, k.subdoc, info.tid),
             "",
             annotate_doc(mini_new_doc),
             "..."]
    return "\n".join(lines)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus(args, verbose=True)
    tcache = TimestampCache()
    output_dir = get_output_dir(args, default_overwrite=True)

    for key in corpus:
        print(key)
        new_doc = corpus[key]
        old_doc = copy.deepcopy(new_doc)
        span = _split_dialogue(tcache, new_doc, args.turn)
        diffs = _mini_diff(key, args, old_doc, new_doc, span)
        print("\n".join(diffs).encode('utf-8'), file=sys.stderr)
        save_document(output_dir, key, new_doc)
        commit_info = CommitInfo(key=key,
                                 before=old_doc,
                                 after=new_doc,
                                 span=span,
                                 tid=args.turn)
    announce_output_dir(output_dir)
    if commit_info and not args.no_commit_msg:
        print("-----8<------")
        print(commit_msg(commit_info))
