"""
educe setup: educe is a library for managing and navigating a
variety of discourse corpora
"""

from setuptools import setup, find_packages
import glob
import os
import sys

PY3 = sys.version > '3'

REQS = \
    ['funcparserlib' if PY3 else 'funcparserlib == 0.3.6',
     'pydot' if PY3 else 'pydot == 1.0.28',
     'python-graph-core',
     'python-graph-dot',
     'sh',
     'nltk == 3.0.0'] +\
    (['fuzzy == 1.0'] if not PY3 else [])


setup(name='educe',
      version='0.2',
      author='Eric Kow',
      author_email='eric@erickow.com',
      packages=find_packages(),
      scripts=[f for f in glob.glob('scripts/*') if not os.path.isdir(f)],
      install_requires=REQS)
