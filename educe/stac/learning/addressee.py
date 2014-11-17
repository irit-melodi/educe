#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
EDU addressee prediction
"""

from itertools import takewhile


def is_punct(token):
    "True if the token is tagged as punctuation"
    return token.tag == ','


def is_emoticon(token):
    "True if the token is tagged as an emoticon"
    return token.tag == 'E'


def guess_addressees(current, edu):
    """
    return a set of possible addressees for the given EDU
    or None if unclear

    At the moment, the basis for our guesses is very crude:
    we simply guess that we have an addresee if the EDU ends
    or starts with their name
    """
    context = current.contexts[edu]
    players = {x.lower(): x for x in current.players}  # orig case
    interesting = [x.word.lower() for x in context.tokens
                   if not (is_punct(x) or is_emoticon(x))]
    if not interesting:
        return None

    is_player = lambda x: x in players
    players_prefix = list(takewhile(is_player, interesting))
    players_suffix = list(takewhile(is_player, reversed(interesting)))

    if players_prefix:
        return frozenset(players[x] for x in players_prefix)
    elif players_suffix:
        return frozenset(players[x] for x in players_suffix)
    else:
        return None
