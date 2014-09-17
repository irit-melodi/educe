[![Build Status](https://secure.travis-ci.org/kowey/educe.png)](http://travis-ci.org/kowey/educe)

## About

Educe is a library for working with discourse-annotated corpora.
It also includes some utility scripts for building, maintaining,
and for querying these corpora. Currently supported corpora are

* (SDRT) [STAC][stac] corpus
* (RST) RST Discourse Treebank (experimental, 2014-07-14)
* (PDTB) Penn Discourse Treebank (experimental, 2014-07-14)

If you have a discourse-annotated corpus, or are trying to build one,
you may find it useful to add support for it to educe.

## Installation

First, try

    pip --version


We require pip 1.5 or greater. If your version of pip is older, you
should check with your package management software, or barring that,
try:

    pip install pip --upgrade

If you don't have pip installed, and you don't have a package
manager to install it with, download this [setup
script][setup-distribute] and run

    python distribute_setup.py
    easy_install pip

Finally, if you have the right version of pip installed, then install
educe and its dependencies:

    pip install -r requirements.txt

## See also

* [Documentation][docs]
* [Attelo][attelo]: a discourse parser

[attelo]: http://github.com/kowey/attelo
[setup-distribute]: http://python-distribute.org/distribute_setup.py
[stac]:  http://www.irit.fr/STAC/
[glozz]: http://www.glozz.org/
[docs]:  https://educe.readthedocs.org/en/latest/api-doc/educe.html
