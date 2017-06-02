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

from frozendict import frozendict


class LexEntry(namedtuple("LexEntry",
                          "word lex_class pos subclass")):
    "a single entry in the lexicon"

    def __new__(cls, word, lex_class, pos, subclass):
        pos = pos if pos != '??' else None
        subclass = subclass or None
        return super(LexEntry, cls).__new__(
            cls, word, lex_class, pos, subclass)

    @classmethod
    def read_entry(cls, line):
        """
        Return a LexEntry given the string corresponding to an entry,
        or raise an exception if we can't parse it
        """
        fields = line.split(':')
        if len(fields) == 4:
            [word, lex_class, pos, subclass] = fields
            return cls(word, lex_class, pos, subclass or None)
        elif len(fields) == 3:
            [word, lex_class, pos] = fields
            return cls(word, lex_class, pos, None)
        else:
            oops = "Sorry, I didn't understand this lexicon entry: %s" % line
            raise Exception(oops)

    @classmethod
    def read_entries(cls, items):
        """
        Return a list of LexEntry given an iterable of entry strings, eg. the
        stream for the lines in a file. Blank entries are ignored
        """
        return [cls.read_entry(x.strip()) for x in items
                if len(x.strip()) > 0]


class LexClass(namedtuple("LexClass",
                          ["word_to_subclass",
                           "subclass_to_words"])):
    """
    Grouping together information for a single lexical class.
    Our assumption here is that a word belongs to at most one
    subclass
    """
    @classmethod
    def new_writable_instance(cls):
        """
        A brand new (empty) lex class
        """
        return cls({}, defaultdict(set))

    @classmethod
    def freeze(cls, other):
        """
        A frozen copy of a lex class
        """
        return LexClass(frozendict(other.word_to_subclass.items()),
                        frozendict((k, frozenset(v)) for k, v in
                                   other.subclass_to_words.items()))

    def just_subclasses(self):
        """
        Any subclasses associated with this lexical class
        """
        return frozenset(self.subclass_to_words.keys())

    def just_words(self):
        """
        Any words associated with this lexical class
        """
        return frozenset(self.word_to_subclass.keys())


class Lexicon(namedtuple("Lexicon", "entries")):
    """
    All entries in a wordclass lexicon along with some helpers
    for convenient access

    :param word_to_subclass: class to word to subclass nested dict
    :type word_to_subclass: Dict String (Dict String String)

    :param subclasses_to_words: class to subclass (to words)
    :type subclasses_to_words:  Dict String (Set String)
    """
    @classmethod
    def read_file(cls, filename):
        """
        Read the lexical entries in the file of the given name
        and return a Lexicon

        :: FilePath -> IO Lexicon
        """
        maps = defaultdict(LexClass.new_writable_instance)

        with codecs.open(filename, 'r', 'utf-8') as stream:
            entries = LexEntry.read_entries(stream)

            for ent in entries:
                lclass = maps[ent.lex_class]
                lclass.word_to_subclass[ent.word] = ent.subclass
                lclass.subclass_to_words[ent.subclass].add(ent.word)
            return cls({k: LexClass.freeze(v) for k, v in maps.items()})

    def dump(self):
        """
        Print a lexicon's contents to stdout
        """
        for key, lclass in self.entries.items():
            print(key)
            print(len(str(key)) * '=')
            print()
            print("subclasses")
            print("----------")
            print()
            for subclass, words in lclass.subclass_to_words.items():
                wlist = " ".join(sorted(words))
                print(u"{subclass}\t[{words}]".format(subclass=subclass,
                                                      words=wlist))
            print()
            print("words")
            print("-----")
            print()
            for word, subclass in lclass.word_to_subclass.items():
                print(u"{word}\t[{subclass}]".format(word=word,
                                                     subclass=subclass))


if __name__ == "__main__":
    Lexicon.read_file(sys.argv[1]).dump()
