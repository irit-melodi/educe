#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
EDU addressee prediction
"""

from itertools import takewhile

from educe.stac.annotation import is_edu
from educe.stac.util.context import Context
from .features import players_for_doc


def is_punct(token):
    "True if the token is tagged as punctuation"
    return token.tag == ','


def is_emoticon(token):
    "True if the token is tagged as an emoticon"
    return token.tag == 'E'


def is_verb(token):
    "True if the token is tagged as a verb"
    return token.tag == 'V'


def is_preposition(token):
    "True if the token is tagged as a preposition"
    return token.tag == 'P'


def _is_maybe(pred, lst, idx):
    """
    Given a predicate, a list, and an index, return True if
    the item is in the list and the predicate is true for it
    """
    return pred(lst[idx]) if len(lst) > idx else False


def guess_addressees_for_edu(contexts, players, edu):
    """
    return a set of possible addressees for the given EDU
    or None if unclear

    At the moment, the basis for our guesses is very crude:
    we simply guess that we have an addresee if the EDU ends
    or starts with their name
    """
    context = contexts[edu]
    players_from_lc = {x.lower(): x for x in players}  # orig case

    interesting_toks = [x for x in context.tokens
                        if not (is_punct(x) or is_emoticon(x))]
    interesting_words = [x.word.lower() for x in interesting_toks]

    if not interesting_toks:
        return None

    def take_player_prefix(words):
        """Return any player names at the start of a word
        sequence ::

        [String] -> [String]
        """
        return [players_from_lc[x] for x in
                takewhile(lambda x: x in players_from_lc, words)]

    players_prefix = take_player_prefix(interesting_words)
    players_suffix = take_player_prefix(reversed(interesting_words))

    v_after_prefix = _is_maybe(is_verb, interesting_toks,
                               len(players_prefix))
    p_before_suffix = _is_maybe(is_preposition, interesting_toks,
                                len(interesting_toks) -
                                len(players_suffix) - 1)

    if players_prefix and not v_after_prefix:
        return players_prefix
    elif players_suffix and not p_before_suffix:
        return players_suffix
    else:
        return None


def guess_addressees(inputs, key):
    """
    Given a document, return a dictionary from edus to addressee
    set ::

        (FeatureInputs, FileId) -> Dict Unit (Set String)

    Note that we distinguish between addressed to nobody
    (empty set) and no-addressee (None); although in practice this
    distinction may not be particularly useful
    """
    doc = inputs.corpus[key]
    contexts = Context.for_edus(doc, inputs.postags[key])
    players = players_for_doc(inputs.corpus, key.doc)
    edus = filter(is_edu, doc.units)
    return {x: guess_addressees_for_edu(contexts, players, x)
            for x in edus}
