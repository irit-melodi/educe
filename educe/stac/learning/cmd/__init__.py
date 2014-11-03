"""
stac-learning subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import\
    extract,\
    features,\
    weave

SUBCOMMANDS = [extract,
               features,
               weave]
