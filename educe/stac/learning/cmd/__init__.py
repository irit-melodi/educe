"""
stac-learning subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import\
    extract,\
    features,\
    res_nps,\
    weave

SUBCOMMANDS = [extract,
               features,
               res_nps,
               weave]
