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

    pip --help

If this doesn't work, download this [setup script][setup-distribute] and
run

    python distribute_setup.py
    easy_install pip

If you have pip installed, then install educe and its dependencies:

    pip install -r requirements.txt     --use-mirrors .


## See also

* [Documentation][docs]
* [Attelo][attelo]: a discourse parser

[attelo]: http://github.com/kowey/attelo
[setup-distribute]: http://python-distribute.org/distribute_setup.py
[stac]:  http://www.irit.fr/STAC/
[glozz]: http://www.glozz.org/
[docs]:  https://educe.readthedocs.org/en/latest/api-doc/educe.html
