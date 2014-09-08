"""
stac-util subcommands
"""

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

from . import\
    clean_emoticons,\
    count,\
    filter,\
    filter_graph,\
    graph,\
    insert,\
    merge_dialogue,\
    merge_edus,\
    move,\
    nudge,\
    nudge_dialogue,\
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
               filter_graph,
               rename,
               rewrite,
               merge_dialogue,
               merge_edus,
               split_edu,
               nudge,
               nudge_dialogue,
               insert,
               tmp,
               move]
