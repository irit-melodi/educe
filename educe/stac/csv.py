# Author: Eric Kow
# License: BSD3

"""
STAC project CSV files

STAC uses CSV files for some intermediary steps when initially
preparing data for annotation.  We don't expect these to be
useful outside of that particular context
"""

from __future__ import absolute_import
import csv

csv_headers = [ 'ID'
              , 'Timestamp'
              , 'Emitter'
              , 'Resources'
              , 'Buildups'
              , 'Text'
              , 'Annotation'
              , 'Comment'
              ]
"""
Fields used in intermediary CSV format for preprocessing
"""

csv.register_dialect('stac', csv.excel_tab)

def mk_plain_csv_writer(outfile):
    """
    Just writes records in stac dialect
    """
    return csv.writer(outfile, dialect='stac')

def mk_csv_writer(writer):
    """
    Writes dictionaries.
    See `csv_headers` for details
    """
    return csv.DictWriter(writer, csv_headers, dialect='stac')

def mk_csv_reader(infile):
    """
    Reads into dictionaries.
    See `csv_headers` for details
    """
    return csv.DictReader(infile, dialect='stac')
