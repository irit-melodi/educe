#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Eric Kow
# License: CeCILL-B (French BSD3)

"""
Cheap and cheerful lexicon format used in the STAC project.
One entry per line, blanks ignored.  Each entry associates

 * some word with
 * some kind of category (we call this a "lexical class")
 * an optional part of speech (?? if unknown)
 * an optional subcategory blank if none

Here's an example with all four fields

    purchase:VBEchange:VB:receivable
    acquire:VBEchange:VB:receivable
    give:VBEchange:VB:givable

and one without the notion of subclass

    ought:modal:MD:
    except:negation:??:
"""

from __future__ import print_function
from collections import defaultdict, namedtuple
import codecs
import sys


class WordClass(namedtuple("WordClass",
                           "word lex_class pos subclass")):
    "a single entry in the lexicon"

    def __init__(self, word, lex_class, pos, subclass):
        pos = pos if pos != '??' else None
        subclass = subclass if subclass != '' else None
        super(WordClass, self).__init__(word, lex_class, pos, subclass)

    @classmethod
    def read_entry(cls, line):
        """
        Return a WordClass given the string corresponding to an entry,
        or raise an exception if we can't parse it
        """
        fields = line.split(':')
        if len(fields) == 4:
            [word, lex_class, pos, subclass] = fields
            return cls(word, lex_class, pos, subclass)
        elif len(fields) == 3:
            [word, lex_class, pos] = fields
            return cls(word, lex_class, pos, None)
        else:
            oops = "Sorry, I didn't understand this lexicon entry: %s" % line
            raise Exception(oops)

    @classmethod
    def read_entries(cls, items):
        """
        Return a list of WordClass given an iterable of entry strings, eg. the
        stream for the lines in a file. Blank entries are ignored
        """
        return [cls.read_entry(x.strip()) for x in items
                if len(x.strip()) > 0]

    @classmethod
    def read_lexicon(cls, filename):
        """
        Return a list of WordClass given a filename corresponding to a lexicon
        we want to read
        """
        with codecs.open(filename, 'r', 'utf-8') as stream:
            return cls.read_entries(stream)


def class_dict(items):
    """
    Given a list of WordClass, return a dictionary mapping lexical classes
    to words that belong in that class
    """
    res = defaultdict(dict)
    for item in items:
        res[item.lex_class][item.word] = item.subclass
    return res


def dump_lexicon(infile):
    """
    Given a filename, read it as lexicon and print its contents
    """
    lex = WordClass.read_lexicon(infile)
    for key, entry in class_dict(lex).items():
        print("===== ", key, "=====")
        print(entry)
        for word, subclass in entry.items():
            print(u"{word}\t[{subclass}]".format(word=word,
                                                 subclass=subclass))


if __name__ == "__main__":
    dump_lexicon(sys.argv[1])
