"""This module implements a loader and dumper for the Orange data formats

See `http://docs.orange.biolab.si/reference/rst/Orange.data.formats.html`
"""

from __future__ import absolute_import

import codecs
import csv

from .csv_format import KeyGroupWriter
from .keys import HeaderType


class OrangeWriter(KeyGroupWriter):
    """
    A tab-separated variant of the KeyGroupWriter which is closer
    to Orange's native format.

    It supplies extra headers which add substance/purpose information
    about the variables
    """
    def __init__(self, f, keys, dialect=csv.excel_tab, **kwds):
        super(OrangeWriter, self).__init__(f, keys, dialect=dialect, **kwds)

    def writeheader(self):
        """
        Write *three rows* representing the CSV header for the
        KeyGroup object.
        """
        self.writer.writerow(self.keys.csv_headers(HeaderType.NAME))
        self.writer.writerow(self.keys.csv_headers(HeaderType.SUBSTANCE))
        self.writer.writerow(self.keys.csv_headers(HeaderType.PURPOSE))


def _dump_orange_tab_file(X, y, f):
    """Actually do dump.

    If X is already exhausted when this method is first
    called, a StopIteration exception will be raised.
    """
    # use the first instance to get the header
    row0 = X.next()
    # create writer and write header
    csv_quoting = csv.QUOTE_MINIMAL
    keys = row0
    writer = OrangeWriter(f, keys, quoting=csv_quoting)
    writer.writeheader()
    # do not forget to write first instance
    writer.writerow(row0)
    # now the rest of them
    for row in X:
        writer.writerow(row)
    

def dump_orange_tab_file(X, y, f):
    """Dump the dataset in orange tab format.

    X is a generator
    y is currently ignored, the class if present is embedded in X
    (see educe.learning.keys.ClassKeyGroup);
    this is likely to change in the near future
    """
    with codecs.open(f, 'wb') as f:  # 2to3: replace with open()
        try:
            _dump_orange_tab_file(X, y, f)
        except StopIteration:
            # FIXME: I have a nagging feeling that we should properly
            # support this by just printing a CSV header and nothing
            # else, but I'm trying to minimise code paths and for now
            # failing in this corner case feels like a lesser evil :-/
            sys.exit("No features to extract!")
