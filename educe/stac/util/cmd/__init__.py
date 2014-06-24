"""
stac-util subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import\
    clean_emoticons,\
    count,\
    filter,\
    graph,\
    insert,\
    merge_dialogue,\
    move,\
    nudge,\
    rename,\
    rewrite,\
    split_edu,\
    text,\
    tmp

SUBCOMMANDS = [clean_emoticons,
               text,
               count,
               graph,
               filter,
               rename,
               rewrite,
               merge_dialogue,
               split_edu,
               nudge,
               insert,
               tmp,
               move]
