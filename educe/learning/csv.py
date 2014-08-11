# Author: Eric Kow
# License: CeCILL-B (BSD3 like)

"""
CSV helpers for machine learning

We sometimes need tables represented as CSV files, with a few
odd conventions here and there to help libraries like Orange
"""

from __future__ import absolute_import
import csv
import re

csv.register_dialect('educe', csv.excel_tab)


def mk_plain_csv_writer(outfile):
    """
    Just writes records in stac dialect
    """
    return csv.writer(outfile, dialect='educe')


def tune_for_csv(string):
    """
    Given a string or None, return a variant of that string that
    skirts around possibly buggy CSV implementations

    SIGH: some CSV parsers apparently get really confused by
    empty fields
    """
    if string:
        string2 = string
        string2 = re.sub(r'"', r"''", string2)  # imitating PTB slightly
        string2 = re.sub(r',', r'-COMMA-', string2)
        string2 = re.sub(r'\\', r'-BACKSLASH-', string2)
        return string2
    else:
        return '__nil__'


class Utf8DictWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in UTF-8.
    """

    def __init__(self, f, headers, dialect=csv.excel, **kwds):
        b_headers = [s.encode('utf-8') for s in headers]
        self.writer = csv.DictWriter(f, b_headers, dialect=dialect, **kwds)

    def writeheader(self):
        self.writer.writeheader()

    def writerow(self, row):
        def b(x):
            if isinstance(x, basestring):
                return unicode(x).encode('utf-8')
            else:
                return x
        self.writer.writerow({b(k): b(v) for k, v in row.items()})

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


class Utf8DictReader:
    """
    A CSV reader which assumes strings are encoded in UTF-8.
    """

    def __init__(self, f, **kwds):
        self.reader = csv.DictReader(f, **kwds)

    def next(self):
        def u(x):
            if isinstance(x, basestring):
                return unicode(x, 'utf-8')
            else:
                return x

        row = self.reader.next()
        return {u(k): u(v) for k, v in row.items()}

    def __iter__(self):
        return self


class SparseDictReader(csv.DictReader):
    """
    A CSV reader which avoids putting null values in dictionaries
    (note that this is basically a copy of DictReader)
    """
    def __init__(self, f, *args, **kwds):
        csv.DictReader.__init__(self, f, *args, **kwds)

    def next(self):
        if self.line_num == 0:
            # Used only for its side effect.
            self.fieldnames
        row = self.reader.next()
        self.line_num = self.reader.line_num

        # unlike the basic reader, we prefer not to return blanks,
        # because we will typically wind up with a dict full of None
        # values
        while row == []:
            row = self.reader.next()
        d = {}
        for name, col in zip(self.fieldnames, row):
            if len(col) > 0:
                d[name] = col
        return d
