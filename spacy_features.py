#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Loads the feature file written by error_analysis.py, runs spaCy on it, and
extracts a number of linguistic features.

Note that this assumes that the tokens in the .tsv file are in sorted order. In
other words, use a batch size of 1 when running error_analysis.py.

Make sure the packages in requirements_analysis.txt are installed before running
this script.
TODO: what features?
"""

import argparse
import logging
import pandas as pd
import spacy


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-file', '-i', type=str, required=True,
                        help='the tsv file created by error_analysis.py.')
    parser.add_argument('--output-file', '-o', type=str, default=None,
                        help='the tsv file, augmented with linguistic '
                             'features. If not specified, the input file is '
                             'overwritten by this script.')
    parser.add_argument('--model', '-m', type=str, default='en',
                        help='the spaCy model to use. It must have already '
                             'been downloaded prior to running this script. '
                             'The default is "en", which is not necessarily '
                             'be available.')
    parser.add_argument('--log-level', '-L', type=str, default=None,
                        choices=['debug', 'info', 'warning', 'error', 'critical'],
                        help='the logging level.')
    args = parser.parse_args()

    if not args.output_file:
        args.output_file = args.input_file
    return args


def parse_input(text):


def main():
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(process)s - %(levelname)s - %(message)s'
    )
    logging.info(f'Arguments: {args}')

    data = pd.from_csv(args.input_file, sep='\t', header=0)


if __name__ == '__main__':
    main()
