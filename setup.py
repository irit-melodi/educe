from setuptools import setup
import glob
import os

setup(name='educe',
      version='0.2',
      author='Eric Kow',
      author_email='eric.kow@gmail.com',
      packages=['educe',
                'educe.learning',
                'educe.stac',
                'educe.rst_dt',
                'educe.pdtb',
                'educe.external'],
      scripts=[f for f in glob.glob('scripts/*') if not os.path.isdir(f)],
      requires=['python_graph (>= 1.8.2)', 'pydot', 'python_graph_dot']
      )
