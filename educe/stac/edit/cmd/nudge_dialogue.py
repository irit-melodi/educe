# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Slightly adjust dialogue annotation boundaries
"""

from __future__ import print_function
from collections import namedtuple
import copy
import sys

from educe.annotation import Span
import educe.stac as st

from educe.stac.annotation import TurnId
from educe.stac.util.annotate import show_diff, annotate_doc
from educe.stac.util.args import\
    add_usual_input_args, add_usual_output_args,\
    add_commit_args,\
    read_corpus, get_output_dir, announce_output_dir
from educe.stac.util.doc import narrow_to_span
from educe.stac.util.output import save_document


def _mini_diff(k, args, old_doc, new_doc, span):
    """
    Return lines of text to be printed out, showing how the nudge
    affected the text
    """
    mini_old_doc = narrow_to_span(old_doc, span)
    mini_new_doc = narrow_to_span(new_doc, span)
    return ["======= NUDGE TURN {} {} in {} ========".format(
        args.turn, args.direction, k),
            "...",
            show_diff(mini_old_doc, mini_new_doc),
            "...",
            ""]


def _nudge_up(turn, dialogue, next_turn, prev_dialogue):
    """
    Move first turn to previous dialogue (ie. extend the
    previous dialogue to incorporate this turn, and push
    this dialogue to exclude it)

    Return encompassing span to show what we've changed
    """
    if not next_turn:
        sys.exit("Can't move very last turn. "
                 "Try `stac-util merge-dialogue` instead")
    elif not prev_dialogue:
        sys.exit("Can't move from first dialogue. "
                 "Try `stac-util move` instead")
    elif turn.span.char_start - 1 != dialogue.span.char_start:
        sys.exit("Turn %d %s is not at the start of its dialogue %s" %
                 (st.turn_id(turn), turn.span, dialogue.span))

    offset = next_turn.span.char_start - turn.span.char_start
    # take both dialogue boundaries up a bit (to prev turn end)
    prev_dialogue.span.char_end += offset
    dialogue.span.char_start += offset
    return Span.merge_all([prev_dialogue.span, dialogue.span])


def _nudge_down(turn, dialogue, prev_turn, next_dialogue):
    """
    Move last turn to next dialogue. (ie. shorten the right
    boundary of this dialogue and extend the left boundary of
    this dialogue)

    Return encompassing span to show what we've changed
    """
    if not prev_turn:
        sys.exit("Can't move very first turn. "
                 "Try `stac-util merge-dialogue` instead")
    elif not next_dialogue:
        sys.exit("Can't move from last dialogue."
                 "Try `stac-util move` instead")
    elif turn.span.char_end != dialogue.span.char_end:
        sys.exit("Turn %d %s is not at the end of its dialogue %s" %
                 (st.turn_id(turn), turn.span, dialogue.span))

    offset = prev_turn.span.char_end - turn.span.char_end
    # take both dialogue boundaries down a bit (to next turn end)
    next_dialogue.span.char_start += offset
    dialogue.span.char_end += offset
    return Span.merge_all([dialogue.span, next_dialogue.span])


def _window1(pred, annos):
    """Return window of prev, match, next, where match is the item matching
    the predicate.

    All values could be None:

    * prev None if match is the first item
    * match None if no match (prev/next should by rights be None too)
    * next None if match is the last item

    Items must be annotations (we sort on text span)
    """
    now = sorted(annos, key=lambda x: x.text_span())
    before = [None] + now[:-1] + [None]
    after = now[1:] + [None]
    for triplet in zip(before, now, after):
        if pred(triplet[1]):
            return triplet
    return None, None, None


def _nudge_dialogue(doc, tid, direction):
    """
    Move a turn either up or down.
    For feedback purposes, return the span of the affected region
    """
    prev_turn, turn, next_turn = _window1(
        lambda x: st.turn_id(x) == tid,
        [x for x in doc.units if st.is_turn(x)]
    )
    if not turn:
        sys.exit("Could not find turn %d" % tid)

    tspan = turn.text_span()
    prev_dialogue, dialogue, next_dialogue = _window1(
        lambda x: x.text_span().encloses(tspan),
        [x for x in doc.units if st.is_dialogue(x)]
    )

    if direction == "up":
        return _nudge_up(turn, dialogue, next_turn, prev_dialogue)
    elif direction == "down":
        return _nudge_down(turn, dialogue, prev_turn, next_dialogue)
    else:
        raise Exception("Unknown direction " + direction)


# ---------------------------------------------------------------------
#
# ---------------------------------------------------------------------

NAME = 'nudge-dialogue'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    add_usual_input_args(parser, doc_subdoc_required=True)
    add_usual_output_args(parser, default_overwrite=True)
    add_commit_args(parser)
    parser.add_argument('turn', metavar='TURN', type=TurnId.from_string,
                        help='turn number')
    parser.add_argument('direction', choices=["up", "down"],
                        help='move turn up or down')
    parser.set_defaults(func=main)


CommitInfo = namedtuple("CommitTuple", "key before after span tid direction")


def commit_msg(info):
    """
    Generate a commit message describing the operation
    we just did
    """
    k = info.key
    mini_new_doc = narrow_to_span(info.after, info.span)

    lines = ["{}_{}: move turn {} {}".format(
        k.doc, k.subdoc, info.tid, info.direction),
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
    output_dir = get_output_dir(args, default_overwrite=True)

    for k in corpus:
        print(k)
        new_doc = corpus[k]
        old_doc = copy.deepcopy(new_doc)
        span = _nudge_dialogue(new_doc, args.turn, args.direction)
        diffs = _mini_diff(k, args, old_doc, new_doc, span)
        print("\n".join(diffs).encode('utf-8'), file=sys.stderr)
        save_document(output_dir, k, new_doc)
        commit_info = CommitInfo(key=k,
                                 before=old_doc,
                                 after=new_doc,
                                 span=span,
                                 tid=args.turn,
                                 direction=args.direction)
    announce_output_dir(output_dir)
    if commit_info and not args.no_commit_msg:
        print("-----8<------")
        print(commit_msg(commit_info))
