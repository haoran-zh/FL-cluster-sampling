import argparse
import warnings
import os

class ParserArgs(object):
    """
    arguments to be used in the experiment
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.get_general_parser()

    def get_general_parser(self):
        # general settings
        self.parser.add_argument("--alpha", type=int, default=3, help="alpha value")
        self.parser.add_argument("--plot_folder", type=str, help="plot which folder")

    def get_args(self):
        args = self.parser.parse_args()
        return args