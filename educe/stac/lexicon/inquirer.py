"""Load the Inquirer lexicon.

This code used to live in `educe.stac.learning.features` ; to the best
of my knowledge it is not used anywhere in the current codebase but who
knows?
"""

from collections import defaultdict
import re

from educe.learning.educe_csv_format import SparseDictReader


def read_inquirer_lexicon(inq_txt_file, classes):
    """Read and return the local Inquirer lexicon.

    Parameters
    ----------
    inq_txt_file : string
        Path to the local text version of the Inquirer.

    classes : list of string
        List of classes from the Inquirer that should be included.

    Returns
    -------
    words : dict(string, string)
        Map from each class to its list of words.
    """
    with open(inq_txt_file) as cin:
        creader = SparseDictReader(cin, delimiter='\t')
        words = defaultdict(list)
        for row in creader:
            for k in row:
                word = row["Entry"].lower()
                word = re.sub(r'#.*$', r'', word)
                if k in classes:
                    words[k].append(word)
    return words
