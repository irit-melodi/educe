"""
stac-learning subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import\
    addressee,\
    extract,\
    features,\
    res_nps,\
    weave

SUBCOMMANDS = [addressee,
               extract,
               features,
               res_nps,
               weave]
