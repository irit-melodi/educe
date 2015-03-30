"""
Reporting component of sanity checker
"""

from __future__ import print_function
from enum import Enum
import copy
import os
import re
import xml.etree.cElementTree as ET

from educe.stac import (id_to_path)
from . import html as h

# pylint: disable=too-few-public-methods

# ----------------------------------------------------------------------
# printing/formatting helpers
# ----------------------------------------------------------------------


def html_anno_id(parent, anno, bracket=False):
    """
    Create and return an HTML span parent node displaying
    the local annotation id for an annotation item
    """
    id_str = ('(%s)' if bracket else '%s') % anno.local_id()
    h.span(parent, id_str, attrib={'class': 'annoid'})


def snippet(txt, stop=50):
    """
    truncate a string if it's longer than `stop` chars
    """
    if len(txt) > stop:
        return txt[:stop] + "..."
    else:
        return txt

# ---------------------------------------------------------------------
# severity
# ---------------------------------------------------------------------


class Severity(Enum):
    """
    Severity of a sanity check error block
    """
    warning = 1
    error = 2

# ---------------------------------------------------------------------
# report
# ---------------------------------------------------------------------


class HtmlReport(object):
    """
    Representation of a report that we would like to generate.
    Output will be dumped to a directory
    """
    css = """
.annoid  { font-family: monospace; font-size: small; }
.feature { font-family: monospace; }
.snippet { font-style: italic; }
.indented { margin-left:1em; }
.hidden   { display:none; }
.naughty  { color:red;  }
.spillover { color:red; font-weight: bold; } /* needs help to be visible */
.missing  { color:red;  }
.excess   { color:blue; }
"""

    javascript = """
function has(xs, x) {
    for (e in xs) {
       if (xs[e] === x) { return true; }
    }
    return false;
}


function toggle_hidden(name) {
    var ele = document.getElementById(name);
    var anc = document.getElementById('anc_' + name);
    if (has(ele.classList, "hidden")) {
        ele.classList.remove("hidden");
        anc.innerText = "[hide]";
    } else {
        ele.classList.add("hidden");
        anc.innerText = "[show]";
   }
}
"""

    def __init__(self, anno_files, output_dir):
        self.subreports = {}
        self.subreport_sections = {}
        self.subreport_started = {}
        self.anno_files = anno_files
        self.output_dir = output_dir
        self._has_errors = {}  # has non-warnings

    @classmethod
    def mk_output_path(cls, odir, k, extension=''):
        """
        Generate a path within a parent directory, given a
        fileid
        """
        relpath = id_to_path(k)
        ofile_dirname = os.path.join(odir, os.path.dirname(relpath))
        ofile_basename = os.path.basename(relpath)
        return os.path.join(ofile_dirname, ofile_basename) + extension

    def write(self, k, path):
        """
        Write the subreport for a given key to the path.
        No-op if we don't have a sub-report for the given key
        """
        if k in self.subreports:
            htree = self.subreports[k]
            with open(path, 'w') as fout:
                print(ET.tostring(htree, encoding='utf-8'), file=fout)

    def delete(self, k):
        """
        Delete the subreport for a given key.
        This can be used if you want to iterate through lots of different
        keys, generating reports incrementally and then deleting them to
        avoid building up memory.

        No-op if we don't have a sub-report for the given key
        """
        if k in self.subreports:
            del self.subreports[k]
            del self.subreport_sections[k]

    def _add_subreport_link(self, key, html, sep, descr):
        """
        Add a link to some side visualisation for the given subreport
        (eg. graph)
        """
        key2 = copy.copy(key)
        key2.stage = 'discourse'
        rel_p = self.mk_output_path('../../../', key, '.svg')
        h.span(html, sep)
        h.elem(html, 'a', descr, href=rel_p)

    def mk_or_get_subreport(self, k):
        """
        Initialise and cache the subreport for a key, including the
        subreports for each severity level below it

        If already cached, retrieve from cache
        """
        if k in self.subreports:
            return self.subreports[k]

        htree = ET.Element('html')
        self.subreports[k] = htree
        self.subreport_sections[k] = {}
        self.subreport_started[k] = {}

        hhead = h.elem(htree, 'head')
        h.elem(hhead, 'style', text=self.css,
               type='text/css')
        h.elem(hhead, 'script', text=self.javascript,
               type='text/javascript')
        h.elem(htree, 'h1', text=str(k))

        hlinks = h.span(htree)
        h.elem(hlinks, 'i', text=self.anno_files[k][0])
        if k.stage == 'discourse':
            self._add_subreport_link(k, hlinks, ' | ', 'graph')

        h.elem(htree, 'hr')

        # placeholder sections for each severity level,
        # most severe first (ensure error comes before warnings)
        for sev in reversed(list(Severity)):
            self.subreport_sections[k][sev] = h.elem(htree, 'div')
            self.subreport_started[k][sev] = False

        return htree

    def subreport_path(self, k, extension='.report.html'):
        """
        Report for a single document
        """
        return self.mk_output_path(self.output_dir, k, extension)

    def flush_subreport(self, k):
        """
        Write and delete (to save memory)
        """
        html_path = self.subreport_path(k)
        if os.path.exists(html_path):
            os.remove(html_path)  # might be leftover from past check
        self.write(k, html_path)
        self.delete(k)

    # pylint: disable=no-self-use
    def anchor_name(self, k, header):
        """
        HTML anchor name for a report section
        """
        mooshed = (str(k) + ' ' + header).lower()
        mooshed = re.sub(r'[^a-z0-9]+', '_', mooshed)
        return mooshed
    # pylint: disable=no-self-use

    def mk_hidden_with_toggle(self, parent, anchor):
        """
        Attach some javascript and html to the given block-level
        element that turns it into a hide/show toggle block,
        starting out in the hidden state
        """
        h.span(parent, text=' ')
        h.elem(parent, "a",
               text='[show]',
               attrib={'id': 'anc_' + anchor},
               href='#',
               onclick="toggle_hidden('" + anchor + "');")

    # pylint: disable=too-many-arguments
    def report(self, k, err_type, severity, header, items, noisy=False):
        """
        Append bullet points for each item to the appropriate section of
        the appropriate report in progress
        """
        if not items:
            return
        if severity == Severity.error:
            self.set_has_errors(k)

        self.mk_or_get_subreport(k)
        subtree = self.subreport_sections[k][severity]
        # emit a header if we haven't yet done so for this section
        # (avoid showing headers unless we have content for this
        # level of severity)
        if not self.subreport_started[k][severity]:
            sev_str = (severity.name + 's').upper()
            h.elem(subtree, 'h2', text=sev_str)
            self.subreport_started[k][severity] = True

        full_header = err_type + ' ' + severity.name.upper() + ': ' + header
        subdiv = h.elem(subtree, "div")
        h.span(subdiv, text=full_header)

        if noisy:
            self.mk_hidden_with_toggle(subdiv, self.anchor_name(k, header))

        e_ul = h.elem(subdiv, "ul")
        e_ul.attrib['id'] = self.anchor_name(k, header)
        if noisy:
            e_ul.attrib['class'] = 'hidden'

        for item in items:
            e_li = h.elem(e_ul, "li")
            e_li.append(item.html())
            # this is slightly evil: modify the offending annotations
            # with a highlight feature that the educe graphing lib
            # can pick up
            if severity == Severity.error:
                for anno in item.annotations():
                    anno.features["highlight"] = "red"
    # pylint: enable=too-many-arguments

    def set_has_errors(self, k):
        """
        Note that this report has seen at least one error-level
        severity message
        """
        self._has_errors[k] = True

    def has_errors(self, k):
        """
        If we have error-level reports for the given key
        """
        return self._has_errors.get(k, False)

# ---------------------------------------------------------------------
# report item
# ---------------------------------------------------------------------


# pylint: disable=no-self-use
class ReportItem(object):
    """
    An individual reportable entry (usually involves a list of
    annotations), rendered as a block of text in the report
    """
    def annotations(self):
        """
        The annotations which this report item is about
        """
        return []

    def text(self):
        """
        If you don't want to create an HTML visualisation for a
        report item, you can fall back to just generating lines
        of text

        :rtype: [string]
        """
        return []

    def html(self):
        """
        Return an HTML element corresponding to the visualisation
        for this item
        """
        parent = ET.Element('span')
        lines = self.text()
        for line in lines[1:]:
            h.br(parent)
            h.span(parent, line)
        return parent
# pylint: enable=no-self-use


class SimpleReportItem(ReportItem):
    """ Report item which just consists of lines of text
    """
    def __init__(self, lines):
        ReportItem.__init__(self)
        self.lines = lines

    def text(self):
        return self.lines


def mk_microphone(report, k, err_type, severity):
    """
    Return a convenience function that generates report entries
    at a fixed error type and severity level

    :rtype: (string, [ReportItem]) -> string
    """
    def inner(header, items, noisy=False):
        """
        (string, [ReportItem]) -> string
        """
        return report.report(k, err_type, severity, header, items, noisy)
    return inner
