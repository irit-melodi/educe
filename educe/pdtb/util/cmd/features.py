# Author: Eric Kow
# License: CeCILL-B (French BSD3-like)

"""
List features used that would be extracted
"""

from __future__ import print_function

from ..features import\
    FeatureInput, RelKeys


NAME = 'features'


def config_argparser(parser):
    """
    Subcommand flags.

    You should create and pass in the subparser to which the flags
    are to be added.
    """
    parser.set_defaults(func=main)


def main(_):
    """
    Subcommand main.

    You shouldn't need to call this yourself if you're using
    `config_argparser`
    """
    inputs = FeatureInput(corpus=None,
                          debug=True)
    print(RelKeys(inputs).help_text())
