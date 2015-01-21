#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Emit a list of known features
"""

from __future__ import print_function

from ..args import add_usual_input_args

NAME = 'features'

# ----------------------------------------------------------------------
# options
# ----------------------------------------------------------------------


def config_argparser(parser):
    """
    Subcommand flags.
    """
    add_usual_input_args(parser)
    parser.set_defaults(func=main)

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------


def main(args):
    "main for feature listing mode"
    print(args.feature_set.PairKeys().help_text())
