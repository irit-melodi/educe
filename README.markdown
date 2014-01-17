[![Build Status](https://secure.travis-ci.org/kowey/educe.png)](http://travis-ci.org/kowey/educe)

## About

Abstract representation of a discourse-annotated corpus.

Currently, this library is geared towards the needs of the [STAC
research project][stac].  In particular, it assumes

- Settlers of Catan non-cooperative conversation domain
- Segmented Discourse Representation Theory (SDRT)
- [Glozz annotation platform][glozz]

But we are keeping sight of the longer term objectives of generalising
it to at least a wider range of domains

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

[setup-distribute]: http://python-distribute.org/distribute_setup.py
[stac]:  http://www.irit.fr/STAC/
[glozz]: http://www.glozz.org/
[docs]:  https://educe.readthedocs.org/en/latest/api-doc/educe.html
