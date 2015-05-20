"""
stac-edit subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import (delete_anno,
               insert,
               merge_dialogue,
               merge_edus,
               move,
               nudge,
               nudge_dialogue,
               rename,
               rewrite,
               split_dialogue,
               split_edu)

# at the time of this writing argparse doesn't support a way to group
# subcommands into sections, but maybe we can wait for it to grow such
# a feature, or write our own formatter class, or just abuse the command
# epilog
SUBCOMMAND_SECTIONS =\
    [('Annotation ids',
      [rename,
       delete_anno,
       rewrite]),
     ('Boundaries',
      [merge_dialogue,
       merge_edus,
       split_dialogue,
       split_edu,
       nudge,
       nudge_dialogue]),
     ('Block editing',
      [insert,
       move])]

SUBCOMMANDS = []
for descr, section in SUBCOMMAND_SECTIONS:
    SUBCOMMANDS.extend(section)
