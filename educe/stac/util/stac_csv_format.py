# Author: Eric Kow
# License: BSD3

"""
STAC project CSV files

STAC uses CSV files for some intermediary steps when initially
preparing data for annotation.  We don't expect these to be
useful outside of that particular context
"""

from __future__ import absolute_import
from collections import namedtuple
import csv

_CSV_ID = 'ID'
_CSV_TIMESTAMP = 'Timestamp'
_CSV_EMITTER = 'Emitter'
_CSV_RESOURCES = 'Resources'
_CSV_BUILDUPS = 'Buildups'
_CSV_TEXT = 'Text'
_CSV_ANNOTATION = 'Annotation'
_CSV_COMMENT = 'Comment'


CSV_HEADERS = [_CSV_ID,
               _CSV_TIMESTAMP,
               _CSV_EMITTER,
               _CSV_RESOURCES,
               _CSV_BUILDUPS,
               _CSV_TEXT,
               _CSV_ANNOTATION,
               _CSV_COMMENT]

csv.register_dialect('stac', csv.excel_tab)


class Turn(namedtuple('Turn',
                      ['number',
                       'timestamp',
                       'emitter',
                       'res',
                       'builds',
                       'rawtext',
                       'annot',
                       'comment'])):
    """
    High-level representation of a turn as used in the STAC
    internal CSV files during intake)
    """
    def to_dict(self):
        "csv representation of this turn"

        return {_CSV_ID: self.number,
                _CSV_TIMESTAMP: self.timestamp,
                _CSV_EMITTER: self.emitter,
                _CSV_RESOURCES: self.res,
                _CSV_BUILDUPS: self.builds,
                _CSV_TEXT: self.rawtext,
                _CSV_ANNOTATION: self.annot,
                _CSV_COMMENT: self.comment}


def mk_plain_csv_writer(outfile):
    """
    Just writes records in stac dialect
    """
    return csv.writer(outfile, dialect='stac')


class Utf8DictWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in UTF-8.
    """

    def __init__(self, f, headers, dialect=csv.excel, **kwds):
        b_headers = [s.encode('utf-8') for s in headers]
        self.writer = csv.DictWriter(f, b_headers, dialect=dialect, **kwds)

    def writeheader(self):
        """Write the header"""
        self.writer.writeheader()

    def writerow(self, row):
        """Write a row"""
        def b(x):
            """Get a utf-8 encoded version of a string"""
            if isinstance(x, basestring):
                return unicode(x).encode('utf-8')
            else:
                return x
        self.writer.writerow(dict([(b(k), b(v)) for k, v in row.items()]))

    def writerows(self, rows):
        """Write several rows"""
        for row in rows:
            self.writerow(row)


class Utf8DictReader:
    """
    A CSV reader which assumes strings are encoded in UTF-8.
    """

    def __init__(self, f, **kwds):
        self.reader = csv.DictReader(f, **kwds)

    def next(self):
        """Get the content of the next row as a dict"""
        def u(x):
            """Get a unicode version of a string"""
            if isinstance(x, basestring):
                return unicode(x, 'utf-8')
            else:
                return x

        row = self.reader.next()
        return dict([(u(k), u(v)) for k, v in row.items()])

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


def mk_csv_writer(ofile):
    """
    Writes dictionaries.
    See `CSV_HEADERS` for details
    """
    return Utf8DictWriter(ofile, CSV_HEADERS, dialect='stac')


def mk_csv_reader(infile):
    """
    Assumes UTF-8 encoded files.
    Reads into dictionaries with Unicode strings.

    See `Utf8DictReader` if you just want a generic UTF-8 dict
    reader, ie. not using the stac dialect
    """
    return Utf8DictReader(infile, dialect='stac')
