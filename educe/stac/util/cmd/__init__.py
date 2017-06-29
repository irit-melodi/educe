"""
stac-util subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=redefined-builtin
# (we have a command called filter)
from . import (count,
               count_rfc,
               count_shapes,
               dump,
               filter,
               filter_graph,
               graph,
               text)

# at the time of this writing argparse doesn't support a way to group
# subcommands into sections, but maybe we can wait for it to grow such
# a feature, or write our own formatter class, or just abuse the command
# epilog
SUBCOMMAND_SECTIONS = [
    ('Querying', [
        text,
        count,
        count_rfc,
        count_shapes,
        graph
    ]),
    ('Filters', [
        filter,
        filter_graph
    ]),
    ('Dump', [
        dump,
    ]),
]

SUBCOMMANDS = []
for descr, section in SUBCOMMAND_SECTIONS:
    SUBCOMMANDS.extend(section)
