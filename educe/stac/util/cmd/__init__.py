"""
stac-util subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=redefined-builtin
# (we have a command called filter)
from . import (count, filter, filter_graph, graph, insert,
               merge_dialogue, merge_edus, move, nudge, nudge_dialogue, rename,
               rewrite, split_edu, text, tmp)

# at the time of this writing argparse doesn't support a way to group
# subcommands into sections, but maybe we can wait for it to grow such
# a feature, or write our own formatter class, or just abuse the command
# epilog
SUBCOMMAND_SECTIONS =\
    [('Querying',
      [text,
       count,
       graph,
       filter_graph]),
     ('Editing',
      [filter,
       rename,
       rewrite,
       merge_dialogue,
       merge_edus,
       split_edu,
       nudge,
       nudge_dialogue]),
     ('Advanced editing',
      [insert,
       move,
       tmp])]

SUBCOMMANDS = []
for descr, section in SUBCOMMAND_SECTIONS:
    SUBCOMMANDS.extend(section)
