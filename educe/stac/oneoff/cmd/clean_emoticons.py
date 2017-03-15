# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
Merge emoticon-only EDUs into preceding EDU (one-off cleanup)
"""

from __future__ import print_function
from collections import defaultdict
import copy

import educe.stac
import educe.stac.postag
from educe.stac.context import sorted_first_widest
from educe.stac.graph import EnclosureGraph
from educe.stac.util.annotate import show_diff
from educe.stac.util.doc import retarget
from educe.stac.util.glozz import (TimestampCache, set_anno_author,
                                   set_anno_date)
from educe.stac.util.args import (add_usual_output_args,
                                  read_corpus_with_unannotated,
                                  get_output_dir, announce_output_dir)
from educe.stac.util.output import save_document
from educe.util import add_corpus_filters, fields_without


NAME = 'clean-emoticons'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.add_argument('corpus', metavar='DIR',
                        nargs='?',
                        help='corpus dir')
    # don't allow stage control
    add_corpus_filters(parser, fields=fields_without(["stage"]))
    add_usual_output_args(parser, default_overwrite=True)
    parser.set_defaults(func=main)


def is_just_emoticon(tokens):
    """
    Similar to the one in stac.features, but takes WrappedTokens
    """
    def is_emoticon(tok):
        "is emoticon-like"
        return tok.features["tag"] == 'E'\
            or tok.features["word"] in ["(!)"]

    return len(tokens) == 1 and is_emoticon(tokens[0])


def has_links(doc, edu):
    """
    True if this edu is pointed to by any relations or included
    in any CDUs
    """
    return\
        any(x.source == edu or x.target == edu for x in doc.relations) or\
        any(edu in x.members for x in doc.schemas)


def sorted_turns(doc):
    """
    Turn annotations in a document, sorted by text span
    """
    return sorted_first_widest(x for x in doc.units if educe.stac.is_turn(x))


def absorb_emoticon(doc, stamp, penult, last):
    """
    Given a timestamp, and two edus, @penult@ (the second to last edu
    in a turn annotation), and @last@ (an emoticon-only edu that follows it),
    absorb the latter into the former.

    This only mutates `penult` (and updates the timestamp generator), and
    does not return anything

    Note that we also have to update any relations/schemas in the document
    telling them to point to the annotation with the new id
    """
    old_id = penult.local_id()
    penult.span = penult.text_span().merge(last.text_span())
    set_anno_date(penult, stamp)
    set_anno_author(penult, "stacutil")
    retarget(doc, old_id, penult)


def turns_with_final_emoticons(doc, tags):
    """
    Return a tuple of lists.

    Both lists contain the turns in a document that end with the
    pattern EDU emoticon-only-EDU.

    The first (main) list contains those that are not pointed to by any
    relations or schema. The second (warnings only) list contains those
    that have relations or schema pointing to them.

    The reason we distinguish between the two lists is that we don't
    want to touch those in the latter (out of conservatism, the idea
    of removing these from their relations, CDUs seems scary), but we
    want to know about them.
    """
    egraph = EnclosureGraph(doc, tags)
    affected_free_turns = []
    affected_linked_turns = []

    for turn in sorted_turns(doc):
        edus = sorted_first_widest(egraph.inside(turn))

        last_edu = edus[-1]
        if len(edus) > 1 and is_just_emoticon(egraph.inside(last_edu)):
            if has_links(doc, last_edu):
                affected_linked_turns.append(turn)
            else:
                affected_free_turns.append(turn)

    return affected_free_turns, affected_linked_turns


def merge_final_emoticons(tcache, turn_spans, doc, tags):
    """
    Given a timestamp cache and some text spans identifying
    turns with final emoticons in them, and a document:

    1. find the specified turns in the document
    2. absorb their emoticon EDUs into the one before it

    This modifies the document and does not return
    anything
    """
    egraph = EnclosureGraph(doc, tags)
    for turn in sorted_turns(doc):
        if turn.text_span() not in turn_spans:
            continue
        edus = sorted_first_widest(egraph.inside(turn))
        assert len(edus) > 1

        stamp = tcache.get(educe.stac.turn_id(turn))
        last_edu = edus[-1]
        penult_edu = edus[-2]
        absorb_emoticon(doc, stamp, penult_edu, last_edu)
        doc.units.remove(last_edu)


def family_banner(doc, subdoc, keys):
    """
    Header announcing the family we're working on
    """

    def show_member(k):
        "single key in the family"
        if k.annotator:
            return "%s/%s" % (k.annotator, k.stage)
        else:
            return k.stage

    fam = "%s [%s]" % (doc, subdoc)
    members = ", ".join(show_member(x) for x in keys)

    return "========== %s =========== (%s)" % (fam, members)


def main(args):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    corpus = read_corpus_with_unannotated(args)
    postags = educe.stac.postag.read_tags(corpus, args.corpus)
    tcache = TimestampCache()
    output_dir = get_output_dir(args, default_overwrite=True)

    families = defaultdict(list)
    # 2017-03-15 merge emoticon EDUs if *all* (selected?) 'discourse'
    # annotators agree that they are not part of any schema or relation
    discourse_subcorpus = defaultdict(list)
    for k in corpus:
        fam = (k.doc, k.subdoc)
        families[fam].append(k)
        if k.stage == 'discourse':
            discourse_subcorpus[fam].append(k)

    for fam in sorted(families):
        print(family_banner(fam[0], fam[1], families[fam]))
        disc_ks = discourse_subcorpus[fam]

        turns = set()
        warn_turns = set()
        for disc_k in disc_ks:
            doc = corpus[disc_k]
            turns_k, warn_turns_k = turns_with_final_emoticons(
                doc, postags[disc_k])
            turns &= set(turns_k)
            warn_turns |= set(warn_turns_k)
        turns = sorted_first_widest(turns)
        warn_turns = sorted_first_widest(warn_turns)

        warnings = []
        if warn_turns:
            warnings.append("Note: These turns have emoticon-only EDUs that "
                            "I dare not touch because they either "
                            "participate in relations or CDUs: ")
            warnings.extend(" " + doc.text(x.text_span()) for x in warn_turns)
            warnings.append("If the "
                            "relations can be removed, or the CDUs reduced, "
                            "please do this by hand and re-run the script:")

        if not turns:
            warnings.append("Skipping %s (and related); no offending emoticons"
                            % disc_k)

        print("\n".join(warnings))

        if not turns:
            continue

        turn_spans = [x.text_span() for x in turns]
        for k in families[fam]:
            doc = copy.deepcopy(corpus[k])
            tags = postags[k]
            merge_final_emoticons(tcache, turn_spans, doc, tags)
            if k == discourse_subcorpus[fam]:
                for turn_span in turn_spans:
                    print(show_diff(corpus[k], doc, span=turn_span))
                    print()
            save_document(output_dir, k, doc)
        tcache.reset()
    announce_output_dir(output_dir)
