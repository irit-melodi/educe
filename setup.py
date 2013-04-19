from distutils.core import setup
setup(name='educe',
      version='0.1',
      author='Eric Kow',
      author_email='eric.kow@gmail.com',
      packages=['educe', 'educe.stac'],
      scripts=['glozz-dump'],
      requires=['python_graph (>= 1.8.2)', 'pydot', 'python_graph_dot']
      )
