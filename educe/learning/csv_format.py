"""This module implements a loader and dumper for the educe CSV data format
"""

from __future__ import absolute_import

import codecs
import csv


def mk_csv_writer(keys, fstream):
    """
    start off csv writer for a given mode
    """
    csv_quoting = csv.QUOTE_MINIMAL
    writer = KeyGroupWriter(fstream, keys, quoting=csv_quoting)
    writer.writeheader()
    return writer


class KeyGroupWriter(object):
    """
    A CSV writer which will write rows to CSV file "f".
    Enforced UTF-8 encoding

    See the Python CSV_ docs on DictWriter. This class
    is meant to resemble that

    .. _CSV: https://docs.python.org/2/library/csv.html
    """

    def __init__(self, f, keys, dialect=csv.excel, **kwds):
        self.keys = keys
        self.writer = csv.writer(f, dialect=dialect, **kwds)

    def writeheader(self):
        """
        Write a row representing the CSV header for the
        KeyGroup object.
        """
        self.writer.writerow(self.keys.csv_headers(HeaderType.OLD_CSV))

    def writerow(self, row):
        """
        Write a row of KeyGroup values. The values must be
        convertible have a Unicode text representation
        (via the Python `unicode` function). The row will be
        encoded as a UTF-8 bytestring.
        """
        def bytestr(val):
            "bytestring representation of an arbitary value"
            if isinstance(val, basestring):
                return unicode(val).encode('utf-8')
            else:
                return val
        self.writer.writerow([bytestr(cv) for cv in row.csv_values()])

    def writerows(self, rows):
        """
        Write out a sequence of rows
        """
        for row in rows:
            self.writerow(row)
