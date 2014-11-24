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
    interesting = [x.word.lower() for x in context.tokens
                   if not (is_punct(x) or is_emoticon(x))]
    if not interesting:
        return None

    is_player = lambda x: x in players_from_lc
    players_prefix = list(takewhile(is_player, interesting))
    players_suffix = list(takewhile(is_player, reversed(interesting)))

    if players_prefix:
        return frozenset(players_from_lc[x] for x in players_prefix)
    elif players_suffix:
        return frozenset(players_from_lc[x] for x in players_suffix)
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
