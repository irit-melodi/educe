"""Command line options for learning commands"""

from __future__ import print_function
import argparse

from . import (features_li2014,
               features as features_eyk,
               features_dev)


FEATURE_SETS = {'li2014': features_li2014,
                'eyk': features_eyk,
                'dev': features_dev}

DEFAULT_FEATURE_SET = 'li2014'


class FeatureSetAction(argparse.Action):
    """Select the desired feature set"""

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        if nargs is not None:
            raise ValueError("nargs not allowed")
        super(FeatureSetAction, self).__init__(option_strings, dest, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, FEATURE_SETS[values])


def add_usual_input_args(parser):
    """
    Augment a subcommand argparser with typical input arguments.
    Sometimes your subcommand may require slightly different output
    arguments, in which case, just don't call this function.
    """
    parser.add_argument('--feature_set',
                        action=FeatureSetAction,
                        default=FEATURE_SETS[DEFAULT_FEATURE_SET],
                        choices=FEATURE_SETS,
                        help='feature set')
