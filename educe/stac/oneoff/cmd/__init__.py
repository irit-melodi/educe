"""
stac-oneoff subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

# pylint: disable=redefined-builtin
# (we have a command called filter)
from . import (clean_emoticons,
               clean_schemas,
               clean_dialogue_acts)

# at the time of this writing argparse doesn't support a way to group
# subcommands into sections, but maybe we can wait for it to grow such
# a feature, or write our own formatter class, or just abuse the command
# epilog
SUBCOMMAND_SECTIONS =\
    [('Cleanups',
      [clean_emoticons,
       clean_schemas,
       clean_dialogue_acts])]

SUBCOMMANDS = []
for descr, section in SUBCOMMAND_SECTIONS:
    SUBCOMMANDS.extend(section)
